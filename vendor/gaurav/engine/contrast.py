"""Contrast under text, estimated from the JSON without rendering.

For every text element the layers drawn below it are composited at a small
grid of points inside its painted (ink) box: the canvas is white, a shape
contributes its fill colour at its fill alpha times its opacity (ellipses
only inside the ellipse), an image contributes the asset's own pixels
mapped through its `fit` (contain / cover / fill; a missing asset is the
renderer's grey placeholder). Nothing is drawn, no asset pixel is changed:
the asset files are read once, downsampled in memory and sampled.

The text colour (and every style-run colour that covers a meaningful share
of the content) is blended over each point at its own alpha and compared
with the point (WCAG ratio); the element's contrast is the 25th percentile
of those ratios, so a headline whose left half sits on a dark photo and
whose right half spills onto the white canvas is judged by the bad half.
`busy` is the spread of the composited colours under the text (0..255
scale, comparable with `ImageStat.stddev`): text on a quiet panel is
easier to read than text on a gradient of the same mean luminance.

Rivals report 0.99 correlation between this estimate and a real render;
`eval/contrast_check.py` measures ours. numpy comes with skia (cr-renderer)
and is used for the arithmetic; without it the estimate is simply
unavailable (empty dict) and the scorer stays neutral, so the engine never
depends on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.geometry import Box

try:  # numpy ships with skia-python (a cr-renderer dependency)
    import numpy as np
except Exception:  # pragma: no cover - only without the renderer stack
    np = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "brand_kit" / "assets"

GRID = (32, 8)              # sample points across x down the text's ink box (long side first)
PERCENTILE = 25.0           # a text's contrast is this percentile of the point ratios
SLICES = 4                  # ... and never better than the worst of this many slices along the long side (v5)
RUN_MIN_SHARE = 0.10        # style-run colours covering less of the content are ignored
ASSET_MAX_SIDE = 192        # assets are downsampled to this many px (longer side) for sampling
PLACEHOLDER_RGB = (214.0, 214.0, 214.0)   # renderer/converter/assets.py `_placeholder`
CANVAS_RGB = (255.0, 255.0, 255.0)
FLAT_STD = 10.0             # an asset whose pixels vary less than this (0..255) is a flat colour


@dataclass
class TextContrast:
    ratio: float            # 25th-percentile WCAG ratio of the text colour(s) against what is under it,
                            # capped by the worst slice (so half a line over a photo counts)
    busy: float             # spread of the composited background under the text (0..255)
    mean_lum: float         # mean luminance (0..1) under the text
    ratios: dict[str, float]   # per text colour (main colour first)
    slices: list[float] = None  # type: ignore[assignment]  # 25th-percentile ratio per slice along the long side
    mean_rgb: Any = None        # mean composited colour under the text (r, g, b in 0..255; v10 backdrop signatures)


# ---------------------------------------------------------------------------
# colours


def parse_rgba(color: Any) -> tuple[tuple[float, float, float], float]:
    """`rgba(r,g,b,a)` / `rgb(r,g,b)` -> ((r, g, b), alpha); black when unparsable."""
    if not isinstance(color, str) or "(" not in color:
        return (0.0, 0.0, 0.0), 1.0
    inner = color[color.find("(") + 1: color.rfind(")")]
    parts = [p.strip() for p in inner.split(",")]
    try:
        rgb = tuple(float(p) for p in parts[:3])
        if len(rgb) != 3:
            return (0.0, 0.0, 0.0), 1.0
        alpha = max(0.0, min(1.0, float(parts[3]))) if len(parts) >= 4 else 1.0
    except ValueError:
        return (0.0, 0.0, 0.0), 1.0
    return rgb, alpha  # type: ignore[return-value]


def _lum_channel(c):
    c = c / 255.0
    if np is not None and not isinstance(c, float):
        return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    """Relative luminance; works on a (…, 3) array or an (r, g, b) tuple."""
    if np is not None and not isinstance(rgb, tuple):
        return 0.2126 * _lum_channel(rgb[..., 0]) + 0.7152 * _lum_channel(rgb[..., 1]) + 0.0722 * _lum_channel(rgb[..., 2])
    r, g, b = rgb
    return 0.2126 * _lum_channel(r) + 0.7152 * _lum_channel(g) + 0.0722 * _lum_channel(b)


# ---------------------------------------------------------------------------
# assets


@lru_cache(maxsize=16)
def _asset_array(path: str):
    """(H, W, 4) float32 array in 0..255 (alpha 0..255) of the downsampled
    asset, or None when the file cannot be read."""
    if np is None:
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGBA")
            k = ASSET_MAX_SIDE / max(im.width, im.height)
            if k < 1.0:
                im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32)
    except Exception:
        return None


def asset_is_flat(asset_id: str | None, assets_dir: Path | str | None = None) -> bool:
    """A flat-colour asset (a solid card, a colour block saved as a PNG)
    behaves like a shape: it can hold text. Decided from the asset's own
    pixels; a missing or unreadable asset is not flat."""
    if not asset_id or np is None:
        return False
    p = Path(assets_dir or ASSETS_DIR) / str(asset_id)
    if not p.is_file():
        return False
    arr = _asset_array(str(p))
    if arr is None:
        return False
    alpha = arr[..., 3] > 8
    if alpha.mean() < 0.9:
        return False
    rgb = arr[..., :3][alpha]
    return float(rgb.std(axis=0).mean()) < FLAT_STD


# ---------------------------------------------------------------------------
# compositing


def _grid(box: Box):
    nx, ny = GRID if box.w >= box.h else (GRID[1], GRID[0])
    xs = box.x + (np.arange(nx, dtype=np.float32) + 0.5) * (box.w / nx)
    ys = box.y + (np.arange(ny, dtype=np.float32) + 0.5) * (box.h / ny)
    px, py = np.meshgrid(xs, ys)
    return px.ravel(), py.ravel()


def _slice_index(box: Box):
    """Which of the SLICES slices along the box's long side each grid point
    belongs to (same point order as `_grid`)."""
    nx, ny = GRID if box.w >= box.h else (GRID[1], GRID[0])
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    along = ix if box.w >= box.h else iy
    n = max(nx, ny)
    return (along.ravel() * SLICES // n).astype(np.int32)


def _local(px, py, box: Box, rotation: float):
    """Point coordinates in the element's own frame (0..w, 0..h), undoing a
    rotation about the box centre."""
    if rotation:
        a = math.radians(rotation)
        c, s = math.cos(a), math.sin(a)
        dx, dy = px - box.cx, py - box.cy
        lx = c * dx + s * dy + box.w / 2.0
        ly = -s * dx + c * dy + box.h / 2.0
        return lx, ly
    return px - box.x, py - box.y


def _layer(el: dict[str, Any], px, py, assets_dir: Path):
    """(rgb (N,3), alpha (N,)) this element contributes at the points, or None."""
    box = Box.of(el)
    if box.w <= 0 or box.h <= 0:
        return None
    rot = float(el.get("rotation", 0.0) or 0.0)
    ext = _rotated_extent(box, rot)
    if (px.min() > ext.right or px.max() < ext.x or py.min() > ext.bottom or py.max() < ext.y):
        return None
    lx, ly = _local(px, py, box, rot)
    inside = (lx >= 0.0) & (lx < box.w) & (ly >= 0.0) & (ly < box.h)
    if not inside.any():
        return None
    opacity = max(0.0, min(1.0, float(el.get("opacity", 1.0) if el.get("opacity") is not None else 1.0)))
    n = px.shape[0]
    if el["type"] == "shape":
        rgb, fa = parse_rgba(el.get("fill_color"))
        kind = str(el.get("shape_type", "rectangle") or "rectangle")
        if kind == "ellipse":
            u = (lx / box.w - 0.5) * 2.0
            v = (ly / box.h - 0.5) * 2.0
            inside = inside & (u * u + v * v <= 1.0)
        elif kind == "line":
            rgb, fa = parse_rgba(el.get("stroke_color") or el.get("fill_color"))
            sw = max(1.0, float(el.get("stroke_width", 1) or 1))
            inside = inside & (np.abs(ly - box.h / 2.0) <= sw / 2.0 + 0.5)
        col = np.empty((n, 3), dtype=np.float32)
        col[:] = rgb
        return col, np.where(inside, fa * opacity, 0.0).astype(np.float32)
    if el["type"] == "image":
        path = Path(assets_dir) / str(el.get("asset_id") or "")
        arr = _asset_array(str(path)) if path.is_file() else None
        if arr is None:
            col = np.empty((n, 3), dtype=np.float32)
            col[:] = PLACEHOLDER_RGB
            return col, np.where(inside, opacity, 0.0).astype(np.float32)
        ih, iw = arr.shape[0], arr.shape[1]
        fit = str(el.get("fit", "contain") or "contain")
        if fit == "cover":
            s = max(box.w / iw, box.h / ih)
            ix = (lx + (iw * s - box.w) / 2.0) / s
            iy = (ly + (ih * s - box.h) / 2.0) / s
        elif fit == "contain":
            s = min(box.w / iw, box.h / ih)
            ix = (lx - (box.w - iw * s) / 2.0) / s
            iy = (ly - (box.h - ih * s) / 2.0) / s
            inside = inside & (ix >= 0.0) & (ix < iw) & (iy >= 0.0) & (iy < ih)
        else:
            ix = lx / box.w * iw
            iy = ly / box.h * ih
        xi = np.clip(ix.astype(np.int32), 0, iw - 1)
        yi = np.clip(iy.astype(np.int32), 0, ih - 1)
        pix = arr[yi, xi]
        return pix[:, :3], np.where(inside, pix[:, 3] / 255.0 * opacity, 0.0).astype(np.float32)
    return None


def _rotated_extent(box: Box, rotation: float) -> Box:
    if not rotation:
        return box
    a = math.radians(rotation)
    w = box.w * abs(math.cos(a)) + box.h * abs(math.sin(a))
    h = box.w * abs(math.sin(a)) + box.h * abs(math.cos(a))
    return Box(box.cx - w / 2.0, box.cy - h / 2.0, w, h)


def background_under(box: Box, layers: list[dict[str, Any]], assets_dir: Path | str | None = None):
    """Composited colour (N, 3) at the grid points of `box` from `layers`
    (already in drawing order, bottom first). None without numpy."""
    if np is None:
        return None
    px, py = _grid(box)
    col = np.empty((px.shape[0], 3), dtype=np.float32)
    col[:] = CANVAS_RGB
    adir = Path(assets_dir or ASSETS_DIR)
    for el in layers:
        if el["type"] == "text":
            continue
        got = _layer(el, px, py, adir)
        if got is None:
            continue
        rgb, a = got
        a3 = a[:, None]
        col = a3 * rgb + (1.0 - a3) * col
    return col


def draw_order(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The renderer's drawing order: z_index ascending, element order for ties."""
    return sorted(elements, key=lambda e: float(e.get("z_index", 0) or 0))


def text_colours(el: dict[str, Any]) -> list[tuple[str, tuple[float, float, float], float]]:
    """(label, rgb, alpha) for the main colour and every style-run colour
    covering at least RUN_MIN_SHARE of the content."""
    n = max(1, len(str(el.get("content", ""))))
    out = [("main", *parse_rgba(el.get("color")))]
    seen = {str(el.get("color"))}
    for run in el.get("style_runs") or []:
        c = run.get("color")
        if not c or str(c) in seen:
            continue
        try:
            share = (min(n, int(run.get("end", 0))) - max(0, int(run.get("start", 0)))) / n
        except (TypeError, ValueError):
            continue
        if share >= RUN_MIN_SHARE:
            seen.add(str(c))
            out.append((f"run:{c}", *parse_rgba(c)))
    return out


def estimate(design: dict[str, Any], ink_boxes: dict[str, Box] | None = None,
             assets_dir: Path | str | None = None, ids: set[str] | None = None) -> dict[str, TextContrast]:
    """Contrast estimate for every text element of `design` (or those in
    `ids`). `ink_boxes` gives the painted box per text id (default: the
    element's box). Empty when numpy is unavailable."""
    if np is None:
        return {}
    order = draw_order(design["elements"])
    out: dict[str, TextContrast] = {}
    for i, el in enumerate(order):
        if el["type"] != "text" or (ids is not None and el["id"] not in ids):
            continue
        if not str(el.get("content", "")).strip():
            continue
        box = (ink_boxes or {}).get(el["id"]) or Box.of(el)
        if box.w <= 0 or box.h <= 0:
            continue
        bg = background_under(box, order[:i], assets_dir)
        if bg is None:
            continue
        lum_bg = luminance(bg)
        ratios: dict[str, float] = {}
        slices: list[float] = []
        sidx = _slice_index(box)
        for label, rgb, alpha in text_colours(el):
            fg = np.empty_like(bg)
            fg[:] = rgb
            fg = alpha * fg + (1.0 - alpha) * bg
            lum_fg = luminance(fg)
            hi, lo = np.maximum(lum_fg, lum_bg), np.minimum(lum_fg, lum_bg)
            r = (hi + 0.05) / (lo + 0.05)
            # Worst region (v5): the 25th percentile over the whole box
            # passes a line whose last third sits on the white canvas; the
            # slices catch it.
            per_slice = [float(np.percentile(r[sidx == s], PERCENTILE)) for s in range(SLICES) if (sidx == s).any()]
            ratios[label] = min(float(np.percentile(r, PERCENTILE)), min(per_slice) if per_slice else float("inf"))
            if label == "main":
                slices = per_slice
        busy = float(bg.std(axis=0).mean())
        out[el["id"]] = TextContrast(ratio=min(ratios.values()), busy=busy,
                                     mean_lum=float(lum_bg.mean()), ratios=ratios, slices=slices,
                                     mean_rgb=tuple(float(v) for v in bg.mean(axis=0)))
    return out


def available() -> bool:
    return np is not None
