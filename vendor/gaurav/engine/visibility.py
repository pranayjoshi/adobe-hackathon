"""Hidden content (v12): every element the viewer saw in the source must
still be visible in the render, not just present in the JSON.

A content element (text ink, picture, the shapes of a button or badge) is
*hidden* when at least HIDDEN_FRAC of its visible box lies under opaque
elements drawn above it (higher z: a solid rectangle or ellipse, a `cover`
image) while the source showed a real part of it (< SOURCE_FRAC covered
there), so a designer's own layering is never a defect. Measured on
v11's eval slice: ~110 of 4,198 layouts drew a hero, logo, badge or text
under a panel that had grown over it (a white card blown up to a story's
full height above the photo it once sat beside).

`hidden_content` is shared by the judge (a hard fail, engine/score.py) and
the repair in `resize.realize` (`reveal`): a covering *panel* is cut back to
the larger side that clears the hidden unit, or, when the cover is not a
panel (or the cut would leave a sliver), the hidden unit is drawn above it
by z order. Sampling only: no pixels, no native code.
"""

from __future__ import annotations

import re
from typing import Any

from engine.analyze import BACKGROUND, DECORATIVE, PANEL, Analysis
from engine.geometry import Box
from engine.textfit import TextMeasurer, default_measurer

HIDDEN_FRAC = 0.9           # an element this much under opaque higher elements is hidden ...
SOURCE_FRAC = 0.75          # ... unless the source already had it at least this covered (the source showed
                            # less than a quarter of it: a photo peeking out from under a card still counts)
OPAQUE = 0.95               # fill alpha x opacity from which a shape / cover image hides what is below
GRID = 9                    # samples per side of the element's visible box
MIN_PANEL_KEEP = 0.25       # a panel cut back to clear content keeps at least this share of its area


def _alpha(color: Any) -> float:
    m = re.match(r"\s*rgba?\(([^)]*)\)", str(color or ""))
    if not m:
        return 1.0
    parts = [p.strip() for p in m.group(1).split(",")]
    try:
        return float(parts[3]) if len(parts) > 3 else 1.0
    except ValueError:
        return 1.0


def _opacity(el: dict[str, Any]) -> float:
    v = el.get("opacity")
    try:
        return 1.0 if v is None else float(v)
    except (TypeError, ValueError):
        return 1.0


def is_cover(el: dict[str, Any]) -> bool:
    """Does `el` hide what is drawn below it (inside its own shape)?"""
    if abs(float(el.get("rotation", 0.0) or 0.0)) > 0.5:
        return False
    if el["type"] == "shape":
        kind = str(el.get("shape_type", "rectangle") or "rectangle")
        if kind not in ("rectangle", "ellipse"):
            return False
        return _alpha(el.get("fill_color")) * _opacity(el) >= OPAQUE
    if el["type"] == "image":
        return el.get("fit") == "cover" and _opacity(el) >= OPAQUE
    return False


def _inside(el: dict[str, Any], x: float, y: float) -> bool:
    b = Box.of(el)
    if not (b.x <= x <= b.right and b.y <= y <= b.bottom):
        return False
    if el["type"] == "shape" and str(el.get("shape_type", "rectangle") or "rectangle") == "ellipse":
        rx, ry = b.w / 2.0, b.h / 2.0
        if rx <= 0 or ry <= 0:
            return False
        return ((x - b.cx) / rx) ** 2 + ((y - b.cy) / ry) ** 2 <= 1.0
    return True


def _visible_box(el: dict[str, Any], m: TextMeasurer) -> Box:
    if el["type"] == "text":
        from engine.rectify import visible_box
        return visible_box(el, Box.of(el), float(el.get("font_size", 0) or 0), m)
    return Box.of(el)


def _z(elements: list[dict[str, Any]]) -> dict[str, float]:
    return {el["id"]: float(el["z_index"]) if el.get("z_index") is not None else float(i)
            for i, el in enumerate(elements)}


def coverage(elements: list[dict[str, Any]], unit_of: dict[str, str],
             ids: list[str] | None = None, m: TextMeasurer | None = None) -> dict[str, tuple[float, str | None]]:
    """element id -> (share of its visible box under opaque higher elements
    from other units, the id covering most of it)."""
    m = m or default_measurer()
    z = _z(elements)
    covers = [el for el in elements if is_cover(el)]
    out: dict[str, tuple[float, str | None]] = {}
    for el in elements:
        if ids is not None and el["id"] not in ids:
            continue
        vb = _visible_box(el, m)
        if vb.w <= 0 or vb.h <= 0:
            continue
        above = [c for c in covers if z[c["id"]] > z[el["id"]]
                 and unit_of.get(c["id"]) != unit_of.get(el["id"]) and Box.of(c).intersection_area(vb) > 0.0]
        if not above:
            out[el["id"]] = (0.0, None)
            continue
        hit = 0
        by: dict[str, int] = {}
        for i in range(GRID):
            for j in range(GRID):
                x = vb.x + (i + 0.5) * vb.w / GRID
                y = vb.y + (j + 0.5) * vb.h / GRID
                for c in above:
                    if _inside(c, x, y):
                        hit += 1
                        by[c["id"]] = by.get(c["id"], 0) + 1
                        break
        top = max(by, key=by.get) if by else None
        out[el["id"]] = (hit / float(GRID * GRID), top)
    return out


def _content_ids(analysis: Analysis, elements: list[dict[str, Any]]) -> list[str]:
    """Elements whose disappearance a viewer would notice: every member of a
    content unit except free decorations (a panel or background is a zone)."""
    present = {el["id"] for el in elements}
    out = []
    for u in analysis.units:
        if u.role in (BACKGROUND, PANEL, DECORATIVE) or not u.is_content:
            continue
        out.extend(e for e in u.ids if e in present)
    return out


def source_coverage(analysis: Analysis, m: TextMeasurer | None = None) -> dict[str, tuple[float, str | None]]:
    cached = getattr(analysis, "source_coverage", None)
    if cached is not None:
        return cached
    src = list(analysis.source_elements.values())
    cov = coverage(src, analysis.element_unit, _content_ids(analysis, src), m)
    analysis.source_coverage = cov   # type: ignore[attr-defined]
    return cov


def hidden_content(analysis: Analysis, elements: list[dict[str, Any]],
                   m: TextMeasurer | None = None) -> list[tuple[str, str, float]]:
    """(element id, covering element id, covered share) for content the
    layout hid that the source showed. Elements that cannot be drawn
    (zero-size) are ignored; a unit counts once, through its most hidden
    text or picture (shapes of a unit count only when it has nothing else)."""
    m = m or default_measurer()
    ids = _content_ids(analysis, elements)
    if not ids:
        return []
    cov = coverage(elements, analysis.element_unit, ids, m)
    src = source_coverage(analysis, m)
    by_id = {el["id"]: el for el in elements}
    worst: dict[str, tuple[str, str, float]] = {}
    for eid in ids:
        frac, top = cov.get(eid, (0.0, None))
        if top is None or frac < HIDDEN_FRAC or src.get(eid, (0.0, None))[0] >= SOURCE_FRAC:
            continue
        unit = analysis.element_unit.get(eid, eid)
        members = [e for e in analysis.unit(unit).ids if e in by_id] if unit in {u.key for u in analysis.units} else [eid]
        has_main = any(by_id[e]["type"] in ("text", "image") for e in members)
        if has_main and by_id[eid]["type"] == "shape":
            continue
        if unit not in worst or frac > worst[unit][2]:
            worst[unit] = (eid, top, frac)
    return list(worst.values())


def reveal(analysis: Analysis, elements: list[dict[str, Any]], target: Box,
           m: TextMeasurer | None = None) -> list[str]:
    """Repair what `hidden_content` reports, in place. A covering panel (or
    background-like zone) is cut back to the larger side that clears the
    hidden element when that keeps >= MIN_PANEL_KEEP of it; otherwise the
    hidden unit is drawn just above its cover (z order). Returns notes."""
    m = m or default_measurer()
    notes: list[str] = []
    by_id = {el["id"]: el for el in elements}
    for _ in range(3):
        hidden = hidden_content(analysis, elements, m)
        if not hidden:
            break
        for eid, cid, _frac in hidden:
            cover = by_id[cid]
            role = analysis.roles.get(cid)
            vb = _visible_box(by_id[eid], m)
            cut = None
            if role in (PANEL, BACKGROUND, DECORATIVE) and cover["type"] == "shape":
                cut = _cut_away(Box.of(cover), vb)
            if cut is not None:
                cover["x"], cover["y"], cover["width"], cover["height"] = cut.x, cut.y, cut.w, cut.h
                notes.append(f"{cid} cut back to clear {eid}")
                continue
            unit = analysis.element_unit.get(eid, eid)
            members = [e for e in (analysis.unit(unit).ids if unit in {u.key for u in analysis.units} else [eid])
                       if e in by_id]
            z = _z(elements)
            base = z[cid]
            lo = min(z[e] for e in members)
            for e in members:
                by_id[e]["z_index"] = base + 0.5 + (z[e] - lo) * 1e-3
            notes.append(f"{unit} drawn above {cid}")
    if any(isinstance(el.get("z_index"), float) and not float(el["z_index"]).is_integer() for el in elements):
        _renumber(elements)
    return notes


def _renumber(elements: list[dict[str, Any]]) -> None:
    """z_index is an integer in the schema: the new stacking order (ties in
    list order, like the renderer's stable sort) becomes ranks 0..n-1."""
    z = _z(elements)
    order = sorted(range(len(elements)), key=lambda i: (z[elements[i]["id"]], i))
    for rank, i in enumerate(order):
        elements[i]["z_index"] = rank


def _cut_away(panel: Box, hole: Box) -> Box | None:
    """The largest of the four sub-rectangles of `panel` that do not touch
    `hole` (left, right, above, below), if it keeps MIN_PANEL_KEEP of it."""
    options = [
        Box(panel.x, panel.y, max(0.0, hole.x - panel.x), panel.h),
        Box(hole.right, panel.y, max(0.0, panel.right - hole.right), panel.h),
        Box(panel.x, panel.y, panel.w, max(0.0, hole.y - panel.y)),
        Box(panel.x, hole.bottom, panel.w, max(0.0, panel.bottom - hole.bottom)),
    ]
    best = max(options, key=lambda b: b.area)
    if best.area < MIN_PANEL_KEEP * panel.area or best.w < 1.0 or best.h < 1.0:
        return None
    return best
