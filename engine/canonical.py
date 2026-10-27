"""Canonical geometry helpers shared by every engine stage.

All coordinates are canvas pixels. `Box` is an axis-aligned rectangle; rotated
elements are analysed through their conservative axis-aligned extents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "brand_kit" / "assets"
FONTS_DIR = ROOT / "brand_kit" / "fonts"

# Fields the engine is allowed to change. Everything else is an immutable snapshot.
MUTABLE_FIELDS = {"x", "y", "width", "height", "font_size"}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def union(self, other: "Box") -> "Box":
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        return Box(x0, y0, x1 - x0, y1 - y0)

    def intersection_area(self, other: "Box") -> float:
        w = min(self.x1, other.x1) - max(self.x, other.x)
        h = min(self.y1, other.y1) - max(self.y, other.y)
        return w * h if w > 0 and h > 0 else 0.0

    def clip(self, cw: float, ch: float) -> "Box":
        x0, y0 = max(0.0, self.x), max(0.0, self.y)
        x1, y1 = min(cw, self.x1), min(ch, self.y1)
        return Box(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "w": round(self.w, 2), "h": round(self.h, 2)}


def union_all(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out


def element_box(el: dict[str, Any]) -> Box:
    return Box(float(el["x"]), float(el["y"]), float(el["width"]), float(el["height"]))


def rotated_extent(el: dict[str, Any]) -> Box:
    """Axis-aligned box that contains both the unrotated element box and its
    rotated footprint (rotation is about the element centre)."""
    b = element_box(el)
    rot = float(el.get("rotation") or 0.0)
    if abs(rot) < 1e-6:
        return b
    t = math.radians(rot)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    rw, rh = b.w * c + b.h * s, b.w * s + b.h * c
    rb = Box(b.cx - rw / 2, b.cy - rh / 2, rw, rh)
    return rb.union(b)


# --------------------------------------------------------------------------
# Asset inspection: transparent margins matter for spacing (a logo PNG is
# often mostly empty pixels), so layout reasons about the visible content box.
# --------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _asset_info(asset_id: str) -> tuple[int, int, tuple[int, int, int, int]] | None:
    path = (ASSETS_DIR / asset_id).resolve()
    if not path.is_relative_to(ASSETS_DIR.resolve()):
        return None
    if not path.is_file():
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                bbox = im.convert("RGBA").getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
            else:
                bbox = (0, 0, w, h)
            return w, h, bbox or (0, 0, w, h)
    except Exception:
        return None


@lru_cache(maxsize=256)
def asset_opaque_fraction(asset_id: str) -> float:
    """Share of pixels that are fully opaque (alpha >= 250); 0 when the asset is unreadable."""
    path = (ASSETS_DIR / asset_id).resolve()
    if not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():
        return 0.0
    try:
        from PIL import Image

        with Image.open(path) as im:
            if im.mode not in ("RGBA", "LA") and "transparency" not in im.info:
                return 1.0
            hist = im.convert("RGBA").getchannel("A").histogram()
            return sum(hist[250:]) / max(1, sum(hist))
    except Exception:
        return 0.0


def asset_exists(asset_id: str | None) -> bool:
    return bool(asset_id) and _asset_info(asset_id) is not None


def visual_fraction(el: dict[str, Any]) -> tuple[float, float, float, float]:
    """(left, top, width, height) of the visible pixels as fractions of the
    element box, emulating the renderer's contain/cover/fill behaviour."""
    if el.get("type") != "image":
        return 0.0, 0.0, 1.0, 1.0
    info = _asset_info(el.get("asset_id") or "")
    fit = el.get("fit", "contain")
    if info is None or fit in ("cover", "fill"):
        if info is not None and fit == "fill":
            iw, ih, (l, t, r, b) = info
            return l / iw, t / ih, (r - l) / iw, (b - t) / ih
        return 0.0, 0.0, 1.0, 1.0
    iw, ih, (l, t, r, b) = info
    bw, bh = max(1e-6, float(el["width"])), max(1e-6, float(el["height"]))
    scale = min(bw / iw, bh / ih)
    ox, oy = (bw - iw * scale) / 2, (bh - ih * scale) / 2
    return (
        (ox + l * scale) / bw,
        (oy + t * scale) / bh,
        ((r - l) * scale) / bw,
        ((b - t) * scale) / bh,
    )


def visual_box(el: dict[str, Any]) -> Box:
    if el.get("type") == "image":
        fl, ft, fw, fh = visual_fraction(el)
        b = element_box(el)
        return Box(b.x + fl * b.w, b.y + ft * b.h, fw * b.w, fh * b.h)
    return rotated_extent(el)


def parse_alpha(color: str | None) -> float:
    if not color or "rgba" not in color:
        return 1.0
    try:
        return float(color.split("(")[1].rstrip(")").split(",")[3])
    except Exception:
        return 1.0


def backdrop_patch(element: dict[str, Any], width: float, height: float) -> dict[str, float]:
    """Fill flexible backdrops; uniformly contain original image geometry.

    A background role does not authorize stretching an image. Cover images
    already opt into cropping; other images keep their source aspect ratio.
    """
    if element['type'] != 'image' or element.get('fit') == 'cover':
        return {'x':0.0,'y':0.0,'width':width,'height':height}
    bounds=rotated_extent(element)
    scale=min(width/max(.001,bounds.w),height/max(.001,bounds.h))
    w,h=element['width']*scale,element['height']*scale
    return {'x':(width-w)/2,'y':(height-h)/2,'width':w,'height':h}


def cover_backdrops(source: dict, design: dict) -> dict:
    """Opt-in cover crop for eligible original canvas backgrounds only.

    Uses the same narrow fit exception as invariants/materialization. Ordinary
    images, compound groups, rotations and same-size originals stay untouched.
    """
    import copy
    from .scene import background_cover_allowed
    if source['canvas'] == design['canvas']:
        return design
    out = copy.deepcopy(design)
    originals = {e['id']: e for e in source['elements']}
    for element in out['elements']:
        before = originals[element['id']]
        if background_cover_allowed(source, before, {**element, 'fit': 'cover'}):
            element.update(x=0, y=0, width=out['canvas']['width'],
                           height=out['canvas']['height'], fit='cover')
    return out
