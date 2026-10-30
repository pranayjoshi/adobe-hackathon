"""Layout strategies. Each takes an Analysis + target canvas and returns a
mapping element id -> new Box (plus, for text, a new font size), leaving
everything else about the element untouched.

Currently implemented:
  * `anchor_relayout` — the small-ratio-change path. Every piece of furniture
    keeps its size proportions (uniform scale by the smaller axis factor) and
    slides to keep hugging whatever canvas edge it hugged before. The axis
    that grew more absorbs the extra room as margins and gaps. Because sizes
    scale by min(sx, sy) and positions by the per-axis factor, gaps never
    shrink below their uniformly-scaled size, so no new collisions appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

from engine.analyze import Analysis, BACKGROUND, DECORATIVE, PANEL, PANEL_EDGE, Unit
from engine.geometry import Box, clamp
from engine.textfit import TextMeasurer, default_measurer

CONTENT_MARGIN_FRAC = 0.03   # keep important things at least this far from the edge
PANEL_PAD_FRAC = 0.02        # minimum breathing room between a panel's edge and the content on it


@dataclass
class Placement:
    boxes: dict[str, Box] = field(default_factory=dict)       # element id -> box
    font_scale: dict[str, float] = field(default_factory=dict) # element id -> multiplier
    font_size: dict[str, float] = field(default_factory=dict)  # element id -> explicit size
                                                               # (already fitted; wins over font_scale)
    slots: dict[str, Box] = field(default_factory=dict)        # v10: unit key -> the region a column placer
                                                               # let the unit occupy (engine/backdrop.slide_texts)
    notes: dict[str, Any] = field(default_factory=dict)        # v10: what a strategy did beyond the boxes (slides)
    tracking: dict[str, float] = field(default_factory=dict)   # v11: element id -> letter-spacing multiplier
                                                               # (strip composer: tracking gives way before size)
    leading: dict[str, float] = field(default_factory=dict)    # v11: element id -> line_height (strip composer)


def ratio_change(source_canvas: Box, target: Box) -> float:
    """How different the target shape is: 1.0 = same aspect, 4.3 = square->banner."""
    sx, sy = target.w / source_canvas.w, target.h / source_canvas.h
    return max(sx, sy) / min(sx, sy)


def anchor_relayout(analysis: Analysis, target: Box, content_scale: float | None = None) -> Placement:
    """`content_scale` overrides the uniform size factor (default min(sx, sy));
    candidates use slightly larger factors when the canvas has room, and the
    rectifier/judge sort out what that does to the gaps."""
    src = analysis.canvas
    sx, sy = target.w / src.w, target.h / src.h
    s = min(sx, sy) if content_scale is None else float(content_scale)
    out = Placement()

    deco = decorative_boxes(analysis, target, s)
    for unit in analysis.units:
        if unit.role == BACKGROUND:
            # A background keeps covering the canvas; the renderer's `fit`
            # handles cropping. Non-image backgrounds (big washes) are just
            # stretched to the canvas, which is what a wash wants.
            for eid, mb in unit.members.items():
                out.boxes[eid] = _cover_box(mb, src, target)
            continue

        if unit.role == PANEL:
            continue   # zones are laid out around the content below

        if unit.role == DECORATIVE:
            new = deco.get(unit.key) or _map_decorative(unit.box, src, target, s)
        else:
            new = _map_content(unit, src, target, sx, sy, s)

        _place_members(unit, new, s, out)
        for eid in unit.ids:
            out.font_scale[eid] = s
    place_panels(analysis, out, target, mode="stretch")
    return out


# ---------------------------------------------------------------------------


def _cover_box(mb: Box, src: Box, target: Box) -> Box:
    """Keep a full-bleed element full-bleed, preserving any intentional
    overhang as a fraction of the canvas."""
    lx = mb.x / src.w
    rx = (mb.right - src.w) / src.w
    ty = mb.y / src.h
    by = (mb.bottom - src.h) / src.h
    x0 = min(0.0, lx) * target.w
    x1 = target.w + max(0.0, rx) * target.w
    y0 = min(0.0, ty) * target.h
    y1 = target.h + max(0.0, by) * target.h
    return Box(x0, y0, x1 - x0, y1 - y0)


def decorative_boxes(analysis: Analysis, target: Box, s: float) -> dict[str, Box]:
    """v10: unit key -> box for the decorations that belong to a pattern
    (engine/patterns.py), placed as one arrangement; empty when patterns
    are off. Decorations not in it keep `_map_decorative`."""
    from engine.patterns import pattern_boxes
    try:
        return pattern_boxes(analysis, target, s)
    except Exception:  # pragma: no cover - a pattern is a bonus, never a dependency
        return {}


def _map_decorative(box: Box, src: Box, target: Box, s: float) -> Box:
    """Decoration scales uniformly and keeps its offset from the nearest
    corner, so a circle bleeding off the top-left still bleeds top-left."""
    w, h = box.w * s, box.h * s
    # nearest horizontal edge
    if box.cx <= src.w / 2:
        x = box.x * s
    else:
        x = target.w - (src.w - box.right) * s - w
    if box.cy <= src.h / 2:
        y = box.y * s
    else:
        y = target.h - (src.h - box.bottom) * s - h
    return Box(x, y, w, h)


def _map_content(unit: Unit, src: Box, target: Box, sx: float, sy: float, s: float) -> Box:
    box = unit.box
    w, h = box.w * s, box.h * s

    # Horizontal position
    if unit.anchor_x == "left":
        x = box.x * sx
    elif unit.anchor_x == "right":
        x = target.w - (src.w - box.right) * sx - w
    else:  # center / fluid: keep the centre's fractional position
        x = (box.cx / src.w) * target.w - w / 2

    # Vertical position
    if unit.anchor_y == "top":
        y = box.y * sy
    elif unit.anchor_y == "bottom":
        y = target.h - (src.h - box.bottom) * sy - h
    else:
        y = (box.cy / src.h) * target.h - h / 2

    margin_x = CONTENT_MARGIN_FRAC * target.w
    margin_y = CONTENT_MARGIN_FRAC * target.h
    x = clamp(x, min(margin_x, box.x * s), max(target.w - w - margin_x, 0.0))
    y = clamp(y, min(margin_y, box.y * s), max(target.h - h - margin_y, 0.0))
    return Box(x, y, w, h)


def _place_members(unit: Unit, new: Box, s: float, out: Placement) -> None:
    """Put each member of a unit at its scaled offset inside the unit's new box."""
    for eid, mb in unit.members.items():
        dx = (mb.x - unit.box.x) * s
        dy = (mb.y - unit.box.y) * s
        out.boxes[eid] = Box(new.x + dx, new.y + dy, mb.w * s, mb.h * s)


# ---------------------------------------------------------------------------
# panels: flat colour zones laid out around the content they carried


def place_panels(analysis: Analysis, out: Placement, target: Box, mode: str,
                 measurer: TextMeasurer | None = None) -> None:
    """Place every panel unit after the content.

    mode "stretch" (anchor path): the panel keeps its fractional geometry
    (flat colour has no aspect), then grows if the content it carried in
    the source has slid out from under it.
    mode "wrap" (floor plans): the panel is rebuilt around the content it
    carried, with the source's padding scaled to the new canvas; a panel
    that carried nothing keeps its fractional geometry.
    Either way a side that touched a canvas edge in the source touches it
    again (a split background stays split, a bar stays a bar)."""
    src = analysis.canvas
    m = measurer or default_measurer()
    sx, sy = target.w / src.w, target.h / src.h
    s = min(sx, sy)
    ps = clamp(math.sqrt(target.area / src.area), 0.25, 2.5)   # padding scale
    pad_min = clamp(PANEL_PAD_FRAC * min(target.w, target.h), 2.0, 16.0)
    for unit in analysis.panels():
        sb = unit.box
        eid = unit.ids[0]
        rect = _is_rect(unit, analysis)
        touch = _edges_touched(sb, src)
        carried = [u for u in analysis.carried(unit.key) if all(i in out.boxes for i in u.ids)]
        hull_now = Box.union(_placed_visible(u, out, analysis, m) for u in carried) if carried else None
        base = Box(sb.x * sx, sb.y * sy, sb.w * sx, sb.h * sy)
        if not rect:
            base = _uniform_at_fraction(sb, src, target, s)
        if hull_now is not None:
            hull_src = Box.union(u.box for u in carried)
            pads = (max(0.0, hull_src.x - sb.x) * ps, max(0.0, hull_src.y - sb.y) * ps,
                    max(0.0, sb.right - hull_src.right) * ps, max(0.0, sb.bottom - hull_src.bottom) * ps)
            pads = tuple(max(p, pad_min) for p in pads)
            wrap = Box(hull_now.x - pads[0], hull_now.y - pads[1],
                       hull_now.w + pads[0] + pads[2], hull_now.h + pads[1] + pads[3])
            new = wrap if mode == "wrap" else Box.union([base, wrap])
        else:
            new = base
        if rect:
            new = _stick_edges(new, touch, target)
            new = _clip(new, target)
            obstacles = [_placed_visible(u, out, analysis, m)
                         for u in _strangers(analysis, unit) if all(i in out.boxes for i in u.ids)]
            new = _avoid(new, obstacles, hull_now, pad_min)
        else:
            # Keep the shape's proportions: scale it uniformly to cover `new`.
            f = max(new.w / max(sb.w, 1e-6), new.h / max(sb.h, 1e-6))
            w, h = sb.w * f, sb.h * f
            new = Box(new.cx - w / 2.0, new.cy - h / 2.0, w, h)
        for mid, mb in stack_boxes(unit, new, ps).items():
            out.boxes[mid] = mb
            out.font_scale[mid] = s


def stack_boxes(unit: Unit, new: Box, ps: float) -> dict[str, Box]:
    """Member boxes of a panel unit placed at `new` (the unit's hull). A
    layered stack (v5, `kind == "stack"`) keeps each layer's inset from the
    hull, scaled by `ps`, so an offset double card stays a double card."""
    if len(unit.ids) == 1:
        return {unit.ids[0]: new}
    sb = unit.box
    out: dict[str, Box] = {}
    for mid, mb in unit.members.items():
        l, t = max(0.0, mb.x - sb.x) * ps, max(0.0, mb.y - sb.y) * ps
        r, b = max(0.0, sb.right - mb.right) * ps, max(0.0, sb.bottom - mb.bottom) * ps
        # insets never eat the whole box on a tiny canvas
        fx = min(1.0, max(0.0, new.w - 2.0) / max(l + r, 1e-6)) if l + r > 0 else 1.0
        fy = min(1.0, max(0.0, new.h - 2.0) / max(t + b, 1e-6)) if t + b > 0 else 1.0
        out[mid] = Box(new.x + l * fx, new.y + t * fy, max(1.0, new.w - (l + r) * fx), max(1.0, new.h - (t + b) * fy))
    return out


def grow_panels(analysis: Analysis, elements: list[dict[str, Any]], target: Box,
                measurer: TextMeasurer | None = None) -> None:
    """After the rectifier moved content around: grow each panel (in place,
    on materialized elements) so it still covers the units it carried, with
    a small pad, keeping the canvas edges it touches."""
    by_id = {el["id"]: el for el in elements}
    m = measurer or default_measurer()
    pad = clamp(PANEL_PAD_FRAC * min(target.w, target.h), 2.0, 16.0)
    ps = clamp(math.sqrt(target.area / analysis.canvas.area), 0.25, 2.5)
    for unit in analysis.panels():
        el = by_id.get(unit.ids[0])
        if el is None or not all(i in by_id for i in unit.ids):
            continue
        carried = [u for u in analysis.carried(unit.key) if all(i in by_id for i in u.ids)]
        if not carried:
            continue
        hull = Box.union(_material_visible(u, by_id, m) for u in carried)
        cur = Box.union(Box.of(by_id[i]) for i in unit.ids)
        want = Box.union([cur, Box(hull.x - pad, hull.y - pad, hull.w + 2 * pad, hull.h + 2 * pad)])
        if not _is_rect(unit, analysis):
            f = max(want.w / max(cur.w, 1e-6), want.h / max(cur.h, 1e-6))
            if f <= 1.0 + 1e-6:
                continue
            w, h = cur.w * f, cur.h * f
            want = Box(want.cx - w / 2.0, want.cy - h / 2.0, w, h)
        else:
            touch = _edges_touched(cur, target, tol=0.5 / max(target.w, target.h))
            want = _clip(_stick_edges(want, touch, target), target)
            obstacles = [_material_visible(u, by_id, m)
                         for u in _strangers(analysis, unit) if all(i in by_id for i in u.ids)]
            want = _avoid(want, obstacles, hull, pad)
        for mid, mb in stack_boxes(unit, want, ps).items():
            e = by_id[mid]
            e["x"], e["y"], e["width"], e["height"] = mb.x, mb.y, mb.w, mb.h


def _visible(el: dict[str, Any], box: Box, font: float | None, m: TextMeasurer) -> Box:
    """What the viewer sees of an element at `box`: the painted extent of a
    text (caps to descenders, aligned lines), the drawn extent of anything
    else (rotation about the centre). Panels wrap and avoid *that*, not the
    boxes, which for text are taller and often wider than the glyphs."""
    rot = float(el.get("rotation", 0.0) or 0.0)
    if el["type"] == "text" and str(el.get("content", "")).strip():
        try:
            p = m.painted(el, font if font else float(el["font_size"]), max(box.w, 1.0))
            box = Box(box.x + p.x, box.y + p.top, max(p.w, 1.0), max(p.h, 1.0))
        except Exception:  # pragma: no cover - odd text element
            pass
    if not rot:
        return box
    a = math.radians(rot)
    w = box.w * abs(math.cos(a)) + box.h * abs(math.sin(a))
    h = box.w * abs(math.sin(a)) + box.h * abs(math.cos(a))
    return Box(box.cx - w / 2.0, box.cy - h / 2.0, w, h)


def _placed_visible(unit: Unit, out: Placement, analysis: Analysis, m: TextMeasurer) -> Box:
    """Visible box of a unit in a Placement (fonts from the placement)."""
    parts = []
    for eid in unit.ids:
        el = analysis.source_elements.get(eid)
        b = out.boxes[eid]
        if el is None:
            parts.append(b)
            continue
        font = out.font_size.get(eid)
        if font is None and el["type"] == "text":
            font = float(el["font_size"]) * out.font_scale.get(eid, 1.0)
        parts.append(_visible(el, b, font, m))
    return Box.union(parts)


def _material_visible(unit: Unit, by_id: dict[str, dict[str, Any]], m: TextMeasurer) -> Box:
    """Visible box of a unit in a materialized element list."""
    return Box.union(_visible(by_id[i], Box.of(by_id[i]), None, m) for i in unit.ids)


def _strangers(analysis: Analysis, panel: Unit) -> list[Unit]:
    """Content units a panel must stay clear of: everything it does not
    carry (a lower layer it covered would vanish, a higher one would land on
    a new colour). A unit that only partly sat on the panel in the source is
    not carried, and may lose that partial overlap."""
    return [u for u in analysis.content_units() if u.panel != panel.key]


def _avoid(box: Box, obstacles: list[Box], keep: Box | None, gap: float) -> Box:
    """Cut `box` back from every obstacle it overlaps, choosing per obstacle
    the side cut that keeps the most area while still containing `keep`
    (the content the panel carries). An obstacle that cannot be avoided
    without losing `keep` is left overlapped."""
    for ob in obstacles:
        if not box.overlaps(ob, tol=0.5):
            continue
        cands = [
            Box(box.x, box.y, ob.x - gap - box.x, box.h),                       # keep the left part
            Box(ob.right + gap, box.y, box.right - ob.right - gap, box.h),      # keep the right part
            Box(box.x, box.y, box.w, ob.y - gap - box.y),                       # keep the top part
            Box(box.x, ob.bottom + gap, box.w, box.bottom - ob.bottom - gap),   # keep the bottom part
        ]
        best = None
        for c in cands:
            if c.w < 1.0 or c.h < 1.0:
                continue
            if keep is not None and c.contains_fraction(keep) < 0.999:
                continue
            if best is None or c.area > best.area:
                best = c
        if best is not None:
            box = best
    return box


def _is_rect(unit: Unit, analysis: Analysis) -> bool:
    return analysis.shape_types.get(unit.ids[0], "rectangle") == "rectangle"


def _edges_touched(box: Box, canvas: Box, tol: float = PANEL_EDGE) -> tuple[bool, bool, bool, bool]:
    return (box.x <= tol * canvas.w, box.y <= tol * canvas.h,
            box.right >= (1.0 - tol) * canvas.w, box.bottom >= (1.0 - tol) * canvas.h)


def _stick_edges(box: Box, touch: tuple[bool, bool, bool, bool], target: Box) -> Box:
    x0, y0, x1, y1 = box.x, box.y, box.right, box.bottom
    if touch[0]:
        x0 = min(x0, 0.0)
    if touch[1]:
        y0 = min(y0, 0.0)
    if touch[2]:
        x1 = max(x1, target.w)
    if touch[3]:
        y1 = max(y1, target.h)
    return Box(x0, y0, x1 - x0, y1 - y0)


def _clip(box: Box, target: Box) -> Box:
    x0, y0 = max(0.0, box.x), max(0.0, box.y)
    x1, y1 = min(target.w, box.right), min(target.h, box.bottom)
    return Box(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))


def _uniform_at_fraction(sb: Box, src: Box, target: Box, s: float) -> Box:
    w, h = sb.w * s, sb.h * s
    return Box(sb.cx / src.w * target.w - w / 2.0, sb.cy / src.h * target.h - h / 2.0, w, h)


def _drawn_box(el: dict[str, Any]) -> Box:
    """Axis-aligned extent of an element as drawn (rotation about the centre)."""
    box = Box.of(el)
    rot = float(el.get("rotation", 0.0) or 0.0)
    if not rot:
        return box
    a = math.radians(rot)
    w = box.w * abs(math.cos(a)) + box.h * abs(math.sin(a))
    h = box.w * abs(math.sin(a)) + box.h * abs(math.cos(a))
    return Box(box.cx - w / 2.0, box.cy - h / 2.0, w, h)


# ---------------------------------------------------------------------------
# post-processing shared by all strategies


def resolve_vertical_collisions(analysis: Analysis, placement: Placement, target: Box) -> None:
    """Cheap safety net: if two content units overlap in the target but did not
    overlap in the source, push the later one (in reading order) down, or up
    if there is no room below. Rotated elements are compared by bbox."""
    units = sorted(analysis.content_units(), key=lambda u: u.reading_index)
    boxes = {u.key: Box.union(placement.boxes[eid] for eid in u.ids) for u in units}

    def parts(u: Unit) -> list[Box]:
        # A sibling row counts per part (v5): a chip packed between the
        # row's lines is not "under" the row's hull.
        if u.is_row:
            return [Box.union(placement.boxes[eid] for eid in p.ids) for p in u.parts]
        return [boxes[u.key]]

    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if a.box.overlaps(b.box, tol=1.0):
                continue  # intentional in the source
            ba, bb = boxes[a.key], boxes[b.key]
            if not any(x.overlaps(y, tol=1.0) for x in parts(a) for y in parts(b)):
                continue
            push = ba.bottom - bb.y + 4.0
            if bb.bottom + push <= target.h:
                _shift(b, placement, 0.0, push)
                boxes[b.key] = Box(bb.x, bb.y + push, bb.w, bb.h)
            else:
                pull = bb.bottom - ba.y + 4.0
                if ba.y - pull >= 0:
                    _shift(a, placement, 0.0, -pull)
                    boxes[a.key] = Box(ba.x, ba.y - pull, ba.w, ba.h)


def _shift(unit: Unit, placement: Placement, dx: float, dy: float) -> None:
    for eid in unit.ids:
        b = placement.boxes[eid]
        placement.boxes[eid] = Box(b.x + dx, b.y + dy, b.w, b.h)
