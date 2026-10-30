"""Backdrop awareness (v10): the background as a composition.

The engine lays furniture out on a canvas it treats as blank: a headline
that sat in the glow of a gradient, a quote on the quiet part of a photo,
a label over the dark bottom of a picture are moved to where the plan's
slot puts them, and the relation to what lies *under* them is lost. This
module gives the engine a coarse picture of the backdrop and three ways to
keep that relation:

  feature map   `BackdropMap`: a GRID x GRID map (FINE x FINE samples per
                cell) of luminance, local busyness (spread of the colours)
                and colour, from a render without text when pixels are
                available (`from_image`) and otherwise from the JSON
                composite of the non-text layers (`from_json`, the layer
                compositing of engine/contrast.py: canvas, shapes with
                their fill alpha, image assets sampled through their `fit`).
                Pixels are a bonus, never a dependency (engine/render.py).
  signature     `Signature` of a text in a design: mean luminance under its
                painted box, busyness, *glow* (luminance relative to the
                canvas median: in the bright part of a dark gradient, or
                the dark part of a bright one), hue / saturation and the
                non-text element it lies over. `source_signatures` records
                every text's signature in the source once per design.
  slide         `slide_texts`: in a floor plan / structure layout a text
                whose slot leaves slack moves, within the slot, to where the
                target's backdrop looks most like its source backdrop (the
                relation, not the coordinates), never across another unit
                and never off the slot.
  quiet area    `quiet_position`: the critic's "onto the quiet area" move:
                where on (or around) a picture the text's colour reads best
                (engine/contrast.py's estimate on the candidate itself, so a
                `cover` crop is sampled as it will be rendered).
  judge term    `judge_signatures` / `similarity`: `read.backdrop` in
                engine/score.py compares a candidate's headline / subhead /
                CTA signatures with the source's, from pixels when the
                second judging stage has them, from the JSON composite
                otherwise, like with like so both stages agree in direction.

CHAMELEON_BACKGROUND=0 switches everything here off (byte-identical output
to the v9 engine); CHAMELEON_BACKDROP_SLIDE / _QUIET / _JUDGE / _PATTERNS
switch one piece (PIECE_DEFAULTS: the judge term is off by default, it did
not pay on the designer references, docs/EVAL_V10.md; engine/patterns.py
reads the last).
Nothing here looks at element ids except to carry them through.
"""

from __future__ import annotations

import colorsys
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine import contrast as ce
from engine.analyze import BODY, CTA, HEADLINE, SUBHEAD, TEXT, Analysis
from engine.geometry import Box, clamp

logger = logging.getLogger(__name__)
np = ce.np

FLAG = "CHAMELEON_BACKGROUND"
DEFAULT_ON = "1"                  # decided by the numbers in docs/EVAL_V10.md
PIECES = ("slide", "quiet", "judge", "patterns")
PIECE_DEFAULTS = {"slide": "1", "quiet": "1", "patterns": "1",
                  "judge": "0"}   # read.backdrop never paid on the designer references at weight 1, 2 or 3

GRID = 24                         # cells per side of the map
FINE = 3                          # samples per cell side (busyness is the spread inside a box)
SLIDE_FINE = 2                    # ... for the per-candidate map the slide samples (cost: ~1 ms at 48 x 48)
RENDER_SIZE = 400                 # text-less render used for a pixel map (the renderer's cost is mostly setup)
PIXELS_FLAG = "CHAMELEON_BACKDROP_PIXELS"   # =1: the judge term also compares pixels with pixels (the source is
PIXELS_DEFAULT = "0"              # rendered once per task without text: ~70 ms, measured in docs/EVAL_V10.md)
ASSETS_DIR = ce.ASSETS_DIR

# Signature distance: luminance band, glow (luminance relative to the canvas
# median), busyness (0..255 spread, saturating at BUSY_SCALE) and hue (only
# between saturated colours). D_FULL is the distance at which the judge
# term reaches 0 and MIN_GAIN the improvement a slide must bring.
W_LUM, W_GLOW, W_BUSY, W_HUE = 0.6, 1.0, 0.6, 0.3
LIGHTNESS_GAMMA = 0.45            # relative luminance -> perceptual lightness (about sRGB's curve)
BUSY_SCALE = 60.0
HUE_MIN_SAT = 0.15
D_FULL = 0.6
MIN_GAIN = 0.12
MOVE_COST = 0.25                  # distance units per canvas width / height moved (shortest move wins ties)
SLIDE_STEPS = (7, 5)              # positions tried across the slot's slack (x, y)
SLIDE_ROLES = (HEADLINE, SUBHEAD, BODY, CTA, TEXT)
JUDGE_ROLES = (HEADLINE, SUBHEAD, CTA)
OVER_MIN = 0.5                    # a text lies "over" the non-text element holding this much of its ink
QUIET_STEPS = (5, 5)              # positions tried over the picture by the quiet-area move
QUIET_MIN_GAIN = 1.2              # ... accepted when the contrast ratio improves by this factor (or passes the drop rule)


# ---------------------------------------------------------------------------
# switches


def enabled(piece: str | None = None) -> bool:
    """Is backdrop awareness (or one `piece` of it) on? Read per call so
    tests and the measurement tools can flip it."""
    if os.environ.get(FLAG, DEFAULT_ON).strip().lower() in ("0", "false", "no", "off"):
        return False
    if piece is None:
        return True
    if piece not in PIECES:
        raise ValueError(piece)
    return os.environ.get(f"CHAMELEON_BACKDROP_{piece.upper()}", PIECE_DEFAULTS[piece]).strip().lower() \
        not in ("0", "false", "no", "off")


def pixels_enabled() -> bool:
    """Does the judge term use the second stage's pixels (and render the
    source once per task to compare like with like)? Off by default: the
    JSON composite samples the same asset pixels through the same `fit`,
    and the render costs ~70 ms per task."""
    return enabled("judge") and os.environ.get(PIXELS_FLAG, PIXELS_DEFAULT).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# signatures


@dataclass
class Signature:
    lum: float                    # mean lightness under the box (0..1, `lightness` of the luminance)
    busy: float                   # spread of the colours under it (0..255, engine/contrast.py's `busy`)
    glow: float                   # lum minus the canvas median lightness (-1..1)
    hue: float = -1.0             # hue of the mean colour (degrees), -1 for grey
    sat: float = 0.0              # saturation of the mean colour (0..1)
    over: str | None = None       # id of the non-text element drawn below the text that holds most of its ink

    def as_dict(self) -> dict[str, Any]:
        return {"lum": round(self.lum, 3), "busy": round(self.busy, 1), "glow": round(self.glow, 3),
                "hue": round(self.hue, 0), "sat": round(self.sat, 2), "over": self.over}


def _hue_sat(rgb) -> tuple[float, float]:
    r, g, b = (clamp(float(c) / 255.0, 0.0, 1.0) for c in rgb)
    h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0 if s >= HUE_MIN_SAT else -1.0), s


def lightness(lum: float) -> float:
    """Perceptual lightness (0..1) of a relative luminance: the dark half of
    a gradient spans a few hundredths of luminance and a third of this."""
    return clamp(float(lum), 0.0, 1.0) ** LIGHTNESS_GAMMA


def make_signature(lum: float, busy: float, median: float, rgb=None, over: str | None = None) -> Signature:
    """`lum` and `median` are relative luminances; the signature keeps lightness."""
    hue, sat = _hue_sat(rgb) if rgb is not None else (-1.0, 0.0)
    light, med = lightness(lum), lightness(median)
    return Signature(lum=light, busy=float(busy), glow=light - med, hue=hue, sat=sat, over=over)


def distance(a: Signature, b: Signature) -> float:
    """How different two backdrops look (0 = the same band, glow, texture
    and colour). `over` is not part of it: the slide uses it separately."""
    d = W_LUM * abs(a.lum - b.lum) + W_GLOW * abs(a.glow - b.glow)
    d += W_BUSY * abs(clamp(a.busy / BUSY_SCALE, 0.0, 1.0) - clamp(b.busy / BUSY_SCALE, 0.0, 1.0))
    if a.hue >= 0.0 and b.hue >= 0.0:
        dh = abs(a.hue - b.hue)
        dh = min(dh, 360.0 - dh) / 180.0
        d += W_HUE * min(a.sat, b.sat) * dh
    return d


def similarity(a: Signature, b: Signature) -> float:
    """1 when the backdrops match, 0 at D_FULL."""
    return clamp(1.0 - distance(a, b) / D_FULL, 0.0, 1.0)


# ---------------------------------------------------------------------------
# the map


@dataclass
class BackdropMap:
    """Luminance / colour of the backdrop at FINE x GRID samples per side,
    plus summed-area tables so a box's signature costs O(1)."""

    w: float                      # design px
    h: float
    n: int                        # samples per side
    lum: Any                      # (n, n) float32, 0..1
    rgb: Any                      # (n, n, 3) float32, 0..255
    median: float
    how: str                      # "pixels" | "estimate"

    def __post_init__(self) -> None:
        self._s_lum = _integral(self.lum)
        self._s_rgb = _integral(self.rgb)
        self._s_rgb2 = _integral(self.rgb * self.rgb)
        self._s_n = _integral(np.ones_like(self.lum))

    def _range(self, box: Box) -> tuple[int, int, int, int]:
        """Sample index range [x0, x1) x [y0, y1) covering `box`, at least
        one sample wide in each direction, inside the map."""
        n = self.n
        x0 = int(math.floor(clamp(box.x / self.w, 0.0, 1.0) * n))
        x1 = int(math.ceil(clamp(box.right / self.w, 0.0, 1.0) * n))
        y0 = int(math.floor(clamp(box.y / self.h, 0.0, 1.0) * n))
        y1 = int(math.ceil(clamp(box.bottom / self.h, 0.0, 1.0) * n))
        x0, y0 = min(x0, n - 1), min(y0, n - 1)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        return x0, x1, y0, y1

    def signature(self, box: Box, over: str | None = None) -> Signature:
        x0, x1, y0, y1 = self._range(box)
        cnt = _area(self._s_n, x0, x1, y0, y1)
        if cnt <= 0:
            return make_signature(self.median, 0.0, self.median)
        lum = _area(self._s_lum, x0, x1, y0, y1) / cnt
        mean = _area(self._s_rgb, x0, x1, y0, y1) / cnt
        sq = _area(self._s_rgb2, x0, x1, y0, y1) / cnt
        var = np.maximum(sq - mean * mean, 0.0)
        busy = float(np.sqrt(var).mean())
        return make_signature(float(lum), busy, self.median, mean, over)

    def as_dict(self) -> dict[str, Any]:
        return {"how": self.how, "n": self.n, "median": round(self.median, 3)}


def _integral(a):
    """Summed-area table with a zero row / column in front."""
    s = np.cumsum(np.cumsum(a, axis=0), axis=1)
    pad = [(1, 0), (1, 0)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(s, pad)


def _area(s, x0: int, x1: int, y0: int, y1: int):
    return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]


def _sample_points(w: float, h: float, n: int):
    xs = (np.arange(n, dtype=np.float32) + 0.5) * (w / n)
    ys = (np.arange(n, dtype=np.float32) + 0.5) * (h / n)
    px, py = np.meshgrid(xs, ys)
    return px.ravel(), py.ravel()


def from_json(design: dict[str, Any], assets_dir: Path | str | None = None, fine: int = FINE,
              elements: list[dict[str, Any]] | None = None) -> BackdropMap | None:
    """The backdrop map from the JSON alone: the non-text layers composited
    (engine/contrast.py) at GRID x `fine` samples per side. None without numpy."""
    if np is None:
        return None
    w, h = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    if w <= 0 or h <= 0:
        return None
    n = GRID * max(1, int(fine))
    px, py = _sample_points(w, h, n)
    col = np.empty((px.shape[0], 3), dtype=np.float32)
    col[:] = ce.CANVAS_RGB
    adir = Path(assets_dir or ASSETS_DIR)
    for el in ce.draw_order(elements if elements is not None else design["elements"]):
        if el["type"] == "text":
            continue
        try:
            got = ce._layer(el, px, py, adir)
        except Exception:  # pragma: no cover - odd element geometry
            got = None
        if got is None:
            continue
        rgb, a = got
        a3 = a[:, None]
        col = a3 * rgb + (1.0 - a3) * col
    rgb = col.reshape(n, n, 3)
    lum = ce.luminance(rgb).astype(np.float32)
    return BackdropMap(w, h, n, lum, rgb, float(np.median(lum)), "estimate")


def from_image(img, design: dict[str, Any], fine: int = FINE) -> BackdropMap | None:
    """The backdrop map from a render without text (a PIL image of the design)."""
    if np is None or img is None:
        return None
    try:
        from PIL import Image

        n = GRID * max(1, int(fine))
        small = img.convert("RGB").resize((n, n), Image.BOX)
        rgb = np.asarray(small, dtype=np.float32)
    except Exception:  # pragma: no cover
        return None
    lum = ce.luminance(rgb).astype(np.float32)
    return BackdropMap(float(design["canvas"]["width"]), float(design["canvas"]["height"]), n, lum, rgb,
                       float(np.median(lum)), "pixels")


def canvas_median(design: dict[str, Any], image=None) -> tuple[float, str] | None:
    """(median luminance of the backdrop, how) for a candidate: from the
    image when given (a text-less render), else a coarse JSON composite."""
    m = from_image(image, design, fine=1) if image is not None else None
    if m is None:
        m = from_json(design, fine=1)
    return (m.median, m.how) if m is not None else None


# ---------------------------------------------------------------------------
# what a text lies over


def over_element(ink: Box, elements: list[dict[str, Any]], z_text: float) -> str | None:
    """Id of the non-text element drawn below the text (z < `z_text`) that
    holds at least OVER_MIN of its ink; the topmost such element."""
    order = ce.draw_order(elements)
    best: tuple[float, str] | None = None
    for i, el in enumerate(order):
        if el["type"] == "text":
            continue
        z = float(el.get("z_index", i) if el.get("z_index") is not None else i)
        if z >= z_text:
            continue
        box = Box.of(el)
        rot = float(el.get("rotation", 0.0) or 0.0)
        if rot:
            box = ce._rotated_extent(box, rot)
        on = box.contains_fraction(ink)
        if on >= OVER_MIN and (best is None or z >= best[0]):
            best = (z, el["id"])
    return best[1] if best else None


def _z_of(el: dict[str, Any], index: int) -> float:
    return float(el["z_index"]) if el.get("z_index") is not None else float(index)


# ---------------------------------------------------------------------------
# source signatures (cached per analysis)


def source_map(analysis: Analysis, source: dict[str, Any] | None = None) -> BackdropMap | None:
    """The source's backdrop map from the JSON (cached on the analysis)."""
    cached = getattr(analysis, "_backdrop_map", None)
    if cached is not None or getattr(analysis, "_backdrop_map_tried", False):
        return cached
    analysis._backdrop_map_tried = True   # type: ignore[attr-defined]
    design = source or {"canvas": {"width": analysis.canvas.w, "height": analysis.canvas.h},
                        "elements": list(analysis.source_elements.values())}
    try:
        m = from_json(design)
    except Exception:  # pragma: no cover
        logger.exception("source backdrop map failed")
        m = None
    analysis._backdrop_map = m   # type: ignore[attr-defined]
    return m


def source_signatures(analysis: Analysis, measurer=None) -> dict[str, Signature]:
    """Backdrop signature of every text of the source (map-based: the same
    convention `slide_texts` uses for the target), by text id, cached."""
    cached = getattr(analysis, "_backdrop_src_sigs", None)
    if cached is not None:
        return cached
    out: dict[str, Signature] = {}
    m = source_map(analysis)
    if m is not None:
        from engine.rectify import visible_box
        from engine.textfit import default_measurer

        tm = measurer or default_measurer()
        elements = list(analysis.source_elements.values())
        for i, el in enumerate(elements):
            if el["type"] != "text" or not str(el.get("content", "")).strip():
                continue
            try:
                ink = visible_box(el, Box.of(el), float(el["font_size"]), tm)
            except Exception:  # pragma: no cover
                continue
            out[el["id"]] = m.signature(ink, over_element(ink, elements, _z_of(el, i)))
    analysis._backdrop_src_sigs = out   # type: ignore[attr-defined]
    return out


# ---------------------------------------------------------------------------
# the judge's signatures (per-text estimate / pixel statistics + canvas median)


def judge_signatures(analysis: Analysis, source: dict[str, Any], design: dict[str, Any],
                     stats: dict[str, tuple[float, float, Any]], image=None, measurer=None
                     ) -> tuple[dict[str, Signature], dict[str, Signature], str] | None:
    """(candidate signatures, source signatures, how) for the texts in
    `stats` ({text id: (mean luminance, busyness, mean rgb)} from the
    scorer's contrast estimate or its pixel sampling). With an `image` the
    source is rendered once per task without text and both sides come from
    pixels; without one both come from the JSON composite. None when the
    source cannot be measured the same way (then the judge is neutral)."""
    how = "pixels" if image is not None else "estimate"
    src = _source_judge_sigs(analysis, source, how, measurer)
    if not src:
        return None
    med = canvas_median(design, image)
    if med is None:
        return None
    median = med[0]
    now = {eid: make_signature(lum, busy, median, rgb) for eid, (lum, busy, rgb) in stats.items() if eid in src}
    return now, src, how


def _source_judge_sigs(analysis: Analysis, source: dict[str, Any], how: str, measurer=None) -> dict[str, Signature]:
    key = "_backdrop_judge_" + how
    cached = getattr(analysis, key, None)
    if cached is not None:
        return cached
    out: dict[str, Signature] = {}
    try:
        from engine.rectify import visible_box
        from engine.textfit import default_measurer

        tm = measurer or default_measurer()
        ink = {el["id"]: visible_box(el, Box.of(el), float(el["font_size"]), tm)
               for el in source["elements"] if el["type"] == "text" and str(el.get("content", "")).strip()}
        image = None
        if how == "pixels":
            from engine.render import render_design

            image = render_design(source, max_size=RENDER_SIZE, with_text=False)
            if image is None:
                setattr(analysis, key, out)      # no pixels for the source: the pixel judging stays neutral
                return out
        stats = text_stats(source, ink, image, tm)
        med = canvas_median(source, image)
        if med is not None:
            out = {eid: make_signature(lum, busy, med[0], rgb) for eid, (lum, busy, rgb) in stats.items()}
    except Exception:  # pragma: no cover
        logger.exception("source judge signatures failed")
        out = {}
    setattr(analysis, key, out)
    return out


def text_stats(design: dict[str, Any], ink: dict[str, Box], image=None, measurer=None
               ) -> dict[str, tuple[float, float, Any]]:
    """{text id: (mean luminance, busyness, mean rgb)} under each ink box:
    from the image (a render without text) when given, else from the
    JSON contrast estimate."""
    out: dict[str, tuple[float, float, Any]] = {}
    if image is not None:
        if np is None:
            return out
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        k = image.width / float(design["canvas"]["width"])
        for eid, box in ink.items():
            x0, y0 = int(max(0, math.floor(box.x * k))), int(max(0, math.floor(box.y * k)))
            x1, y1 = int(min(image.width, math.ceil(box.right * k))), int(min(image.height, math.ceil(box.bottom * k)))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = arr[y0:y1, x0:x1].reshape(-1, 3)
            out[eid] = (float(ce.luminance(crop).mean()), float(crop.std(axis=0).mean()), crop.mean(axis=0))
        return out
    est = ce.estimate(design, ink, ids=set(ink))
    for eid, tc in est.items():
        out[eid] = (tc.mean_lum, tc.busy, tc.mean_rgb)
    return out


def backdrop_term(analysis: Analysis, source: dict[str, Any], design: dict[str, Any],
                  stats: dict[str, tuple[float, float, Any]], image=None, measurer=None
                  ) -> tuple[float, dict[str, Any]]:
    """`read.backdrop`: importance-weighted similarity of every headline /
    subhead / CTA backdrop signature to the source's; (1.0, {}) when it
    cannot be measured."""
    got = judge_signatures(analysis, source, design, stats, image, measurer)
    if got is None:
        return 1.0, {}
    now, src, how = got
    num = den = 0.0
    notes: dict[str, Any] = {"how": how}
    for eid, sig in now.items():
        role = analysis.roles.get(eid)
        if role not in JUDGE_ROLES or eid not in src:
            continue
        key = analysis.element_unit.get(eid)
        imp = analysis.unit(key).importance if key is not None else 0.3
        imp = max(imp, 0.05)
        sim = similarity(sig, src[eid])
        notes[eid] = {"sim": round(sim, 3), "now": sig.as_dict(), "src": src[eid].as_dict()}
        num += imp * sim
        den += imp
    return (num / den if den else 1.0), notes


# ---------------------------------------------------------------------------
# the slide: keep the relation, not the coordinates


def placement_map(placement, elements: dict[str, dict[str, Any]], target: Box) -> BackdropMap | None:
    """The target's backdrop map for a Placement: its non-text elements at
    their placed boxes (JSON composite; no render at plan time)."""
    els = []
    for eid, b in placement.boxes.items():
        el = elements.get(eid)
        if el is None or el["type"] == "text":
            continue
        els.append(dict(el, x=b.x, y=b.y, width=b.w, height=b.h))
    design = {"canvas": {"width": target.w, "height": target.h}, "elements": els}
    return from_json(design, fine=SLIDE_FINE, elements=els)


def slide_texts(analysis: Analysis, placement, target: Box, elements: dict[str, dict[str, Any]],
                measurer=None) -> list[dict[str, Any]]:
    """Move each text unit that has a slot (`placement.slots`, set by the
    column placer) to the position inside its slot whose backdrop signature
    is closest to the text's source signature, when that is clearly closer
    than where the plan put it (MIN_GAIN) and the move keeps the unit on
    its slot, off every other content unit (pairs the source overlapped
    may still overlap) and in reading order (the slot is between its
    neighbours). Returns a note per moved unit."""
    slots = getattr(placement, "slots", None)
    if not slots or np is None:
        return []
    from engine.rectify import overlap_rules, visible_box, _pair_key
    from engine.textfit import default_measurer

    m = measurer or default_measurer()
    src_sigs = source_signatures(analysis, m)
    if not src_sigs:
        return []
    tmap = placement_map(placement, elements, target)
    if tmap is None:
        return []
    rules = overlap_rules(analysis, m)
    content = [u for u in analysis.content_units() if all(i in placement.boxes for i in u.ids)]
    frames = {u.key: Box.union(placement.boxes[i] for i in u.ids) for u in content}
    placed_els = {eid: dict(elements[eid], x=b.x, y=b.y, width=b.w, height=b.h)
                  for eid, b in placement.boxes.items() if eid in elements}
    for eid, el in placed_els.items():
        if el["type"] == "text":
            f = placement.font_size.get(eid)
            el["font_size"] = float(f if f is not None else float(elements[eid]["font_size"]) * placement.font_scale.get(eid, 1.0))
    moved: list[dict[str, Any]] = []
    for u in sorted(content, key=lambda u: u.reading_index):
        slot = slots.get(u.key)
        if slot is None or u.role not in SLIDE_ROLES:
            continue
        eid = u.primary or u.ids[0]
        el = placed_els.get(eid)
        if el is None or el["type"] != "text" or float(el.get("rotation", 0.0) or 0.0):
            continue
        sig0 = src_sigs.get(eid)
        if sig0 is None:
            continue
        try:
            ink = visible_box(el, Box.of(el), float(el["font_size"]), m)
        except Exception:  # pragma: no cover
            continue
        frame = frames[u.key]
        order = list(placed_els.values())
        z_text = _z_of(elements[eid], list(elements).index(eid))
        cur = tmap.signature(ink, over_element(ink, order, z_text))
        d0 = _slide_distance(sig0, cur, 0.0, 0.0, target)
        if d0 < MIN_GAIN:
            continue                      # already where it belongs
        xs = _steps(slot.x - frame.x, slot.right - frame.right, SLIDE_STEPS[0])
        ys = _steps(slot.y - frame.y, slot.bottom - frame.bottom, SLIDE_STEPS[1])
        if len(xs) == 1 and len(ys) == 1:
            continue
        best = (d0, 0.0, 0.0, cur)
        for dx in xs:
            for dy in ys:
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    continue
                moved_ink = Box(ink.x + dx, ink.y + dy, ink.w, ink.h)
                sig = tmap.signature(moved_ink, over_element(moved_ink, order, z_text))
                d = _slide_distance(sig0, sig, dx, dy, target)
                if d >= best[0] - 1e-9:
                    continue
                nf = Box(frame.x + dx, frame.y + dy, frame.w, frame.h)
                if nf.x < slot.x - 0.5 or nf.y < slot.y - 0.5 or nf.right > slot.right + 0.5 or nf.bottom > slot.bottom + 0.5:
                    continue
                if any(nf.overlaps(frames[o.key], tol=1.0) and rules.src_frac.get(_pair_key(u.key, o.key), 0.0) <= 0.0
                       for o in content if o is not u):
                    continue
                best = (d, dx, dy, sig)
        if best[0] > d0 - MIN_GAIN:
            continue
        _, dx, dy, sig = best
        for i in u.ids:
            b = placement.boxes[i]
            placement.boxes[i] = Box(b.x + dx, b.y + dy, b.w, b.h)
            if i in placed_els:
                placed_els[i]["x"], placed_els[i]["y"] = b.x + dx, b.y + dy
        frames[u.key] = Box(frame.x + dx, frame.y + dy, frame.w, frame.h)
        moved.append({"unit": u.key, "dx": round(dx, 1), "dy": round(dy, 1), "d0": round(d0, 3), "d": round(best[0], 3),
                      "src": sig0.as_dict(), "now": sig.as_dict()})
    return moved


def _slide_distance(sig0: Signature, sig: Signature, dx: float, dy: float, target: Box) -> float:
    d = distance(sig0, sig)
    if sig0.over is not None and sig.over != sig0.over:
        d += W_GLOW * 0.5             # not on the element it sat on in the source
    return d + MOVE_COST * (abs(dx) / max(target.w, 1.0) + abs(dy) / max(target.h, 1.0))


def _steps(lo: float, hi: float, n: int) -> list[float]:
    """`n` offsets from `lo` to `hi` (0 included), [0] when there is no slack."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 4.0:
        return [0.0]
    vals = {0.0}
    for i in range(n):
        vals.add(lo + (hi - lo) * i / (n - 1))
    return sorted(vals)


# ---------------------------------------------------------------------------
# the critic's quiet-area move


def quiet_position(eid: str, pic_key: str, by_id: dict[str, dict[str, Any]], analysis: Analysis,
                   target: Box, measurer=None) -> tuple[float, float, float, float] | None:
    """Where the text `eid` reads best against the picture `pic_key` in the
    current layout: (dx, dy, ratio before, ratio after) for the unit's move
    to the band with the best estimated contrast for its colour (shortest
    move among equals), on canvas, inside the picture when the text was
    inside it in the source, never onto another content unit. None when no
    position reads better than the current one by QUIET_MIN_GAIN."""
    if np is None:
        return None
    from engine.rectify import overlap_rules, rotated_extent, visible_box, _on_fraction, _pair_key
    from engine.relayout import CONTENT_MARGIN_FRAC
    from engine.score import CONTRAST_DROP, CONTRAST_GOOD
    from engine.textfit import default_measurer

    m = measurer or default_measurer()
    el = by_id.get(eid)
    unit_key = analysis.element_unit.get(eid)
    pic = next((u for u in analysis.units if u.key == pic_key), None)
    if el is None or el["type"] != "text" or unit_key is None or pic is None:
        return None
    unit = analysis.unit(unit_key)
    if not all(i in by_id for i in unit.ids) or not all(i in by_id for i in pic.ids):
        return None
    frame_of = lambda ids: Box.union(rotated_extent(Box.of(by_id[i]), float(by_id[i].get("rotation", 0.0) or 0.0)) for i in ids)
    pb = frame_of(pic.ids)
    tf = frame_of(unit.ids)
    try:
        ink = visible_box(el, Box.of(el), float(el["font_size"]), m)
        src_el = analysis.source_elements[eid]
        ink_src = visible_box(src_el, unit.members[eid], None, m)
    except Exception:  # pragma: no cover
        return None
    inside = _on_fraction([ink_src], [pic.box]) >= 0.8
    design = {"canvas": {"width": target.w, "height": target.h}, "elements": list(by_id.values())}
    now = ce.estimate(design, {eid: ink}, ids={eid}).get(eid)
    if now is None:
        return None
    mg = clamp(CONTENT_MARGIN_FRAC * min(target.w, target.h), 2.0, 24.0)
    region = pb if inside else Box(mg, mg, target.w - 2 * mg, target.h - 2 * mg)
    # ink positions: a grid over the region, the ink box kept inside it
    xs = _grid_positions(region.x, region.right, ink.w, QUIET_STEPS[0])
    ys = _grid_positions(region.y, region.bottom, ink.h, QUIET_STEPS[1])
    rules = overlap_rules(analysis, m)
    others = [(o.key, frame_of(o.ids)) for o in analysis.content_units()
              if o.key != unit_key and all(i in by_id for i in o.ids)]
    best: tuple[tuple[float, float], float, float, float] | None = None
    for x in xs:
        for y in ys:
            dx, dy = x - ink.x, y - ink.y
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            nf = Box(tf.x + dx, tf.y + dy, tf.w, tf.h)
            if nf.x < mg - 0.5 or nf.y < mg - 0.5 or nf.right > target.w - mg + 0.5 or nf.bottom > target.h - mg + 0.5:
                continue
            moved = Box(x, y, ink.w, ink.h)
            if not inside and pb.contains_fraction(moved) < 0.3:
                continue              # the point is to read against the picture
            if any(nf.overlaps(ob, tol=1.0) and rules.src_frac.get(_pair_key(unit_key, ok), 0.0) <= 0.0
                   for ok, ob in others if ok != pic_key):
                continue
            tc = ce.estimate(design, {eid: moved}, ids={eid}).get(eid)
            if tc is None or (tc.slices and _straddles(tc.slices)):
                continue
            key = (-min(tc.ratio, CONTRAST_GOOD), abs(dx) + abs(dy))
            if best is None or key < best[0]:
                best = (key, dx, dy, tc.ratio)
    if best is None:
        return None
    _, dx, dy, ratio = best
    if ratio < now.ratio * QUIET_MIN_GAIN and not (now.ratio < CONTRAST_DROP <= ratio):
        return None
    return dx, dy, now.ratio, ratio


def _straddles(slices: list[float]) -> bool:
    from engine.rectify import is_straddle
    return is_straddle(slices)


def _grid_positions(lo: float, hi: float, size: float, n: int) -> list[float]:
    """`n` left / top positions of a box of `size` inside [lo, hi] (its
    centre when it does not fit)."""
    if hi - lo <= size:
        return [(lo + hi) / 2.0 - size / 2.0]
    return [lo + (hi - lo - size) * i / (n - 1) for i in range(n)]
