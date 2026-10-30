"""Deterministic defect fixer, run on every candidate before it is judged.

Works on a materialized design (elements with final boxes and font sizes)
at the level of the analyzer's units, so a group always moves and scales as
one piece and its member insets stay exactly proportional. Each pass:

  1. images keep their aspect ratio (contain)
  2. groups only ever move or scale as one piece, so member insets stay
     exactly as the strategy laid them out (uniform, or a button squeezed
     around its label by the floor plan)
  3. unwanted overlaps between content units are pushed apart (or the less
     important unit shrinks when nothing can move): overlaps that already
     existed in the source, members of one group, and anything touching the
     background / decoration are left alone
  4. every content unit is fully on canvas with a small margin
  5. near-aligned edges and centres snap to a shared line
  6. nearly even gaps in a stack or a row become exactly even

Overlaps are judged on *visible* boxes: a text element counts as its ink
(the measured wrapped lines, aligned like the renderer aligns them), a
rotated element as the axis-aligned extent of its rotated box. Passes
repeat until nothing changes or `MAX_PASSES` is reached, so it always
terminates. Fonts shrink with their unit and are re-fitted so they still
fit their boxes; `content` / `font_family` are never touched.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from engine import contrast as contrast_est
from engine.analyze import Analysis, BACKGROUND, FINE_PRINT, BADGE, Unit
from engine.geometry import Box, clamp
from engine.textfit import CAP_HEIGHT, DEFAULT_LINE_HEIGHT, TextMeasurer, apply_font_size, default_measurer

MAX_PASSES = 6
EPS = 0.01
OVERLAP_TOL = 0.015         # a pair may overlap this much more (fraction of the smaller box) than in the source
MIN_SHRINK = 0.4            # never shrink a unit below this fraction to solve an overlap
MARGIN_FRAC = 0.015         # on-canvas margin for content, as a fraction of min(W, H)
SNAP_FRAC = 0.008           # snap tolerance, fraction of min(W, H)
GAP_FRAC = 0.01             # minimum visible gap between pushed-apart units
EVEN_GAP_RATIO = 1.6        # gaps closer than this ratio (max/min) are made equal
GRAZE_EM = 0.9              # a source text-on-text overlap thinner than this x the smaller font is a collision
                            # (lines that ran into each other), not a designed lockup: it is not carried over
GRAZE_CONTAINED = 0.8       # ... unless one text's ink lies this much on the other's lines and the two differ
LOCKUP_FONT_RATIO = 1.5     # in size by this factor (a label laid over a big figure) or are the same word (a
                            # shadow copy offset by a few px): a lockup however thin the small text is (v6)
# Layering (v6, from the designer references): what the designer layered on
# purpose is not a collision when it is layered again.
LAYER_SRC_MIN = 0.3         # a text whose ink lay at least this much over a picture / shape in the source was
                            # layered on it (role-level: any text of that role may be layered again) ...
LAYER_MIN = 0.3             # ... when at least this much of its ink is on the picture (not grazing its edge)
HOSTED_INSIDE = 0.8         # a text this much inside a shape / image drawn below it is hosted by it, never a collision
LAYER_GROW = 2.0            # a pair the source layered (one side not text) may overlap up to this x as much
STRADDLE_MIN = 3.0          # a text straddles an edge when one slice of it reads under this ratio ...
STRADDLE_MAX = 4.5          # ... and another slice at or above this ...
STRADDLE_RATIO = 4.0        # ... and the best slice reads at least this many times better than the worst
                            # (a sharp edge, not the glow of a gradient); engine/score.py uses the same rule
STRADDLE_PASSES = 2         # the straddle pass runs in the first passes only (it costs a contrast estimate)


@dataclass
class Live:
    """A unit's current geometry while the rectifier works."""

    unit: Unit
    boxes: dict[str, Box]                 # element id -> current box
    fonts: dict[str, float] = field(default_factory=dict)
    changed: bool = False
    shrunk: float = 1.0                   # cumulative scale applied to solve overlaps

    @property
    def box(self) -> Box:
        return Box.union(self.boxes.values())

    def frame(self, elements: dict[str, dict[str, Any]]) -> Box:
        """Axis-aligned extent of the unit as drawn: rotated members count
        with their rotated box (the renderer rotates about the box centre)."""
        return Box.union(rotated_extent(b, float(elements[eid].get("rotation", 0.0) or 0.0))
                         for eid, b in self.boxes.items())

    @property
    def importance(self) -> float:
        return self.unit.importance


# ---------------------------------------------------------------------------
# visible boxes


def rotated_extent(box: Box, rotation: float) -> Box:
    """Axis-aligned extent of `box` rotated about its centre (what the
    renderer draws for a rotated element)."""
    if not rotation:
        return box
    a = math.radians(rotation)
    w = box.w * abs(math.cos(a)) + box.h * abs(math.sin(a))
    h = box.w * abs(math.sin(a)) + box.h * abs(math.cos(a))
    return Box(box.cx - w / 2.0, box.cy - h / 2.0, w, h)


def ink_box(el: dict[str, Any], box: Box, font_size: float, m: TextMeasurer) -> Box:
    """Where the glyphs of a text element land: the painted extent of the
    wrapped lines (caps to the last baseline / descender), aligned inside
    the box like cr-renderer does. See `TextMeasurer.painted`."""
    p = m.painted(el, font_size, max(box.w, 1.0))
    ink = Box(box.x + p.x, box.y + p.top, max(p.w, 1.0), max(p.h, 1.0))
    return rotated_extent(ink, float(el.get("rotation", 0.0) or 0.0))


def ink_lines(el: dict[str, Any], box: Box, font_size: float | None, m: TextMeasurer) -> list[Box]:
    """One painted box per wrapped line of an (unrotated) text: caps top to
    baseline plus the descender allowance `TextMeasurer.painted` uses,
    each line as wide as its glyphs and aligned like the renderer aligns
    it. The hull of a three-line heading covers its line gaps; a label
    that merely touches its last line is not "inside" it (v6)."""
    size = float(font_size if font_size is not None else el["font_size"])
    p = m.painted(el, size, max(box.w, 1.0))
    lh = float(el.get("line_height", DEFAULT_LINE_HEIGHT) or DEFAULT_LINE_HEIGHT)
    cap = CAP_HEIGHT * size
    last_base = p.top + cap + (p.n_lines - 1) * size * lh
    desc = max(0.0, p.bottom - last_base)
    align = str(el.get("text_align", "left") or "left")
    out = []
    for i, lw in enumerate(p.line_widths):
        lw = max(float(lw), 1.0)
        x = box.x + ((box.w - lw) / 2.0 if align == "center" else (box.w - lw) if align == "right" else 0.0)
        out.append(Box(x, box.y + p.top + i * size * lh, lw, max(1.0, cap + desc)))
    return out


def lines_inside(small: list[Box], big: list[Box]) -> float:
    """Share of the `small` lines' area that lies on the `big` lines."""
    area = sum(b.area for b in small)
    if area <= 0:
        return 0.0
    return sum(s.intersection_area(b) for s in small for b in big) / area


def visible_box(el: dict[str, Any], box: Box, font_size: float | None, m: TextMeasurer) -> Box:
    if el["type"] == "text":
        return ink_box(el, box, font_size if font_size is not None else float(el["font_size"]), m)
    return rotated_extent(box, float(el.get("rotation", 0.0) or 0.0))


def unit_visible_box(live: Live, elements: dict[str, dict[str, Any]], m: TextMeasurer) -> Box:
    return Box.union(visible_box(elements[eid], b, live.fonts.get(eid), m) for eid, b in live.boxes.items())


def unit_visible_parts(live: Live, elements: dict[str, dict[str, Any]], m: TextMeasurer) -> list[Box]:
    """Visible boxes used for collision tests: one per sibling of a row unit
    (the badges of a row pinned to two corners do not "overlap" what lies
    between them), one per element of a host unit (a heading and the bar
    above it do not cover the gap between them), otherwise the unit's one
    visible box (a pill with its label is solid)."""
    if live.unit.is_row:
        out = []
        for part in live.unit.parts:
            ids = [i for i in part.ids if i in live.boxes]
            if ids:
                out.append(Box.union(visible_box(elements[i], live.boxes[i], live.fonts.get(i), m) for i in ids))
        if out:
            return out
    if live.unit.kind == "host":
        return [visible_box(elements[i], b, live.fonts.get(i), m) for i, b in live.boxes.items()]
    return [unit_visible_box(live, elements, m)]


def parts_overlap(a: list[Box], b: list[Box], tol: float = 1.0) -> bool:
    return any(x.overlaps(y, tol=tol) for x in a for y in b)


def parts_overlap_fraction(a: list[Box], b: list[Box]) -> float:
    """Largest overlap fraction over the part pairs (fraction of the smaller part)."""
    return max((overlap_fraction(x, y) for x in a for y in b), default=0.0)


def parts_intersection(a: list[Box], b: list[Box]) -> float:
    return sum(x.intersection_area(y) for x in a for y in b)


# ---------------------------------------------------------------------------
# entry point


def rectify(analysis: Analysis, elements: list[dict[str, Any]], target: Box,
            measurer: TextMeasurer | None = None, passes: int = MAX_PASSES,
            protect: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Return a corrected copy of `elements` (same ids, same order).
    `protect` (v9, the critic's raised texts): unit keys that are never the
    unit that shrinks to resolve an overlap when the other side is free."""
    m = measurer or default_measurer()
    elements = copy.deepcopy(elements)
    by_id = {el["id"]: el for el in elements}
    src_by_id = {eid: el for el in elements for eid in [el["id"]]}  # same dicts; fonts come from here
    lives = _lives(analysis, by_id)
    r = _Rectifier(analysis, by_id, lives, target, m, protect=set(protect))
    for _ in range(max(1, passes)):
        changed = r.run_pass()
        if not changed:
            break
    r.write_back()
    return elements


def _lives(analysis: Analysis, by_id: dict[str, dict[str, Any]]) -> list[Live]:
    out = []
    for u in analysis.units:
        if not u.is_content:
            continue
        boxes = {eid: Box.of(by_id[eid]) for eid in u.ids if eid in by_id}
        if not boxes:
            continue
        fonts = {eid: float(by_id[eid]["font_size"]) for eid in boxes if by_id[eid]["type"] == "text"}
        out.append(Live(u, boxes, fonts))
    return out


class _Rectifier:
    def __init__(self, analysis: Analysis, by_id: dict[str, dict[str, Any]], lives: list[Live],
                 target: Box, m: TextMeasurer, protect: set[str] | None = None):
        self.analysis = analysis
        self.protect = protect or set()
        self.by_id = by_id
        self.lives = sorted(lives, key=lambda l: l.unit.reading_index)
        self.target = target
        self.m = m
        mn = min(target.w, target.h)
        self.margin = clamp(MARGIN_FRAC * mn, 1.0, 16.0)
        self.snap_tol = clamp(SNAP_FRAC * mn, 1.0, 6.0)
        self.gap = clamp(GAP_FRAC * mn, 2.0, 12.0)
        # How much each pair overlapped in the source (visible boxes): a
        # pair may keep that much overlap (plus OVERLAP_TOL); more is a defect
        # unless a layering allowance applies (`OverlapRules`, v6).
        self.rules = overlap_rules(analysis, m)
        self.src_frac = self.rules.src_frac
        self._vis: dict[str, Box] = {}
        self._parts: dict[str, list[Box]] = {}
        self.src_fonts = {eid: float(el["font_size"]) for eid, el in by_id.items() if el["type"] == "text"}
        self.passes_run = 0

    # -- geometry helpers -----------------------------------------------------

    def frame(self, l: Live) -> Box:
        return l.frame(self.by_id)

    def vis(self, l: Live) -> Box:
        b = self._vis.get(l.unit.key)
        if b is None:
            b = unit_visible_box(l, self.by_id, self.m)
            self._vis[l.unit.key] = b
        return b

    def parts(self, l: Live) -> list[Box]:
        p = self._parts.get(l.unit.key)
        if p is None:
            p = unit_visible_parts(l, self.by_id, self.m)
            self._parts[l.unit.key] = p
        return p

    def invalidate(self, l: Live) -> None:
        self._vis.pop(l.unit.key, None)
        self._parts.pop(l.unit.key, None)
        l.changed = True

    def move(self, l: Live, dx: float, dy: float) -> None:
        if abs(dx) < EPS and abs(dy) < EPS:
            return
        for eid, b in l.boxes.items():
            l.boxes[eid] = Box(b.x + dx, b.y + dy, b.w, b.h)
        self.invalidate(l)

    def scale(self, l: Live, f: float, ax: float, ay: float) -> None:
        """Scale the whole unit by `f` about the point (ax, ay); fonts follow."""
        if abs(f - 1.0) < 1e-6:
            return
        for eid, b in l.boxes.items():
            l.boxes[eid] = Box(ax + (b.x - ax) * f, ay + (b.y - ay) * f, b.w * f, b.h * f)
            if eid in l.fonts:
                l.fonts[eid] = max(1.0, l.fonts[eid] * f)
        self.refit_fonts(l)
        self.invalidate(l)

    def refit_fonts(self, l: Live) -> None:
        for eid, font in list(l.fonts.items()):
            el = self.by_id[eid]
            b = l.boxes[eid]
            size = float(int(font)) if font >= 1.0 else font
            if not self.m.fits(el, size, b.w, b.h):
                size = self.m.fit_font_size(el, b.w, b.h, size, 1.0)
            l.fonts[eid] = size

    def allowed_pair(self, a: Live, b: Live) -> bool:
        """Did the pair overlap in the source? Such pairs are only separated
        when their overlap grew beyond what `OverlapRules` allows."""
        return self.src_frac.get(_pair_key(a.unit.key, b.unit.key), 0.0) > 0.0

    def excess(self, a: Live, b: Live, pa: list[Box] | None = None, pb: list[Box] | None = None) -> float:
        """Overlap fraction now minus what the source had (negative: fine)."""
        pa = pa if pa is not None else self.parts(a)
        pb = pb if pb is not None else self.parts(b)
        return parts_overlap_fraction(pa, pb) - self.src_frac.get(_pair_key(a.unit.key, b.unit.key), 0.0)

    def collides(self, l: Live, box: Box | None = None, ignore: Iterable[str] = ()) -> bool:
        """Would `l` (at `box`, default its current visible box) overlap any
        unit it must not overlap?"""
        if box is not None:
            # a hypothetical position: shift this unit's parts by the same offset
            cur = self.vis(l)
            dx, dy = box.x - cur.x, box.y - cur.y
            mine = [Box(p.x + dx, p.y + dy, p.w, p.h) for p in self.parts(l)]
        else:
            mine = self.parts(l)
        skip = set(ignore)
        for o in self.lives:
            if o is l or o.unit.key in skip:
                continue
            if self.rules.conflict(l, o, mine, self.parts(o)):
                return True
        return False

    def on_canvas(self, box: Box) -> bool:
        t, mg = self.target, self.margin
        return box.x >= mg - EPS and box.y >= mg - EPS and box.right <= t.w - mg + EPS and box.bottom <= t.h - mg + EPS

    def inside_canvas_tol(self, box: Box) -> bool:
        t = self.target
        return box.x >= -0.5 and box.y >= -0.5 and box.right <= t.w + 0.5 and box.bottom <= t.h + 0.5

    # -- passes -----------------------------------------------------------------

    def run_pass(self) -> bool:
        for l in self.lives:
            l.changed = False
        self.fix_aspect()
        self.fix_overlaps()
        self.fix_on_canvas()
        if self.passes_run < STRADDLE_PASSES:
            self.fix_straddles()
        self.passes_run += 1
        self.snap()
        self.even_gaps()
        return any(l.changed for l in self.lives)

    # -- straddles (v5) ---------------------------------------------------------

    def _snapshot(self) -> list[dict[str, Any]]:
        """The elements with the units' current geometry (non-content
        elements as they are), for the contrast estimate."""
        cur: dict[str, tuple[Box, float | None]] = {}
        for l in self.lives:
            for eid, b in l.boxes.items():
                cur[eid] = (b, l.fonts.get(eid))
        out = []
        for eid, el in self.by_id.items():
            if eid in cur:
                b, font = cur[eid]
                el = dict(el, x=b.x, y=b.y, width=b.w, height=b.h)
                if font is not None:
                    el["font_size"] = font
            out.append(el)
        return out

    def _source_straddles(self) -> set[str]:
        s = getattr(self.analysis, "source_straddle", None)
        if s is None:
            src = {"canvas": {"width": self.analysis.canvas.w, "height": self.analysis.canvas.h},
                   "elements": list(self.analysis.source_elements.values())}
            ink = {}
            for el in src["elements"]:
                if el["type"] == "text" and str(el.get("content", "")).strip():
                    ink[el["id"]] = visible_box(el, Box.of(el), float(el["font_size"]), self.m)
            est = contrast_est.estimate(src, ink)
            s = {k for k, v in est.items() if _is_straddle(v)}
            self.analysis.source_straddle = s   # type: ignore[attr-defined]
        return s

    def fix_straddles(self) -> None:
        """A text lying across the edge of a photo or a panel (one part on
        the dark picture, the other on the white canvas) is slid fully onto
        the side where it reads, when that spot is free and on canvas. Only
        texts that did not straddle in the source; the contrast comes from
        the JSON estimate (engine/contrast.py), so no pixels are needed."""
        if not contrast_est.available():
            return
        elements = self._snapshot()
        design = {"canvas": {"width": self.target.w, "height": self.target.h}, "elements": elements}
        by_id = {el["id"]: el for el in elements}
        texts: dict[str, Live] = {}
        for l in self.lives:
            if l.unit.role in (FINE_PRINT, BADGE):
                continue
            for eid in l.boxes:
                el = by_id[eid]
                if el["type"] == "text" and str(el.get("content", "")).strip():
                    texts[eid] = l
        if not texts:
            return
        ink = {eid: visible_box(by_id[eid], Box.of(by_id[eid]), float(by_id[eid]["font_size"]), self.m)
               for eid in texts}
        est = contrast_est.estimate(design, ink, ids=set(texts))
        src = self._source_straddles()
        order = contrast_est.draw_order(elements)
        z = {el["id"]: i for i, el in enumerate(order)}
        for eid, tc in est.items():
            if not _is_straddle(tc) or eid in src:
                continue
            l = texts[eid]
            box = ink[eid]
            g = self.gap
            moves: list[tuple[float, float]] = []
            for el in elements:
                if el["type"] == "text" or z[el["id"]] >= z[eid]:
                    continue
                role = self.analysis.roles.get(el["id"])
                if role == BACKGROUND:
                    continue
                ext = rotated_extent(Box.of(el), float(el.get("rotation", 0.0) or 0.0))
                if not ext.overlaps(box, tol=0.0) or ext.contains_fraction(box) >= 0.999:
                    continue
                for edge in (ext.x, ext.right):
                    if box.x < edge < box.right:
                        moves.append((edge - g - box.right, 0.0))   # fully left of the edge
                        moves.append((edge + g - box.x, 0.0))       # fully right of it
                for edge in (ext.y, ext.bottom):
                    if box.y < edge < box.bottom:
                        moves.append((0.0, edge - g - box.bottom))
                        moves.append((0.0, edge + g - box.y))
            best = None
            for dx, dy in moves:
                full = self.frame(l)
                full = Box(full.x + dx, full.y + dy, full.w, full.h)
                if not self.on_canvas(full):
                    continue
                vb = self.vis(l)
                if self.collides(l, Box(vb.x + dx, vb.y + dy, vb.w, vb.h)):
                    continue
                moved = Box(box.x + dx, box.y + dy, box.w, box.h)
                there = contrast_est.estimate(design, {eid: moved}, ids={eid}).get(eid)
                if there is None or _is_straddle(there) or there.ratio < STRADDLE_MIN:
                    continue
                key = (-min(there.ratio, STRADDLE_MAX), abs(dx) + abs(dy))
                if best is None or key < best[0]:
                    best = (key, dx, dy)
            if best is not None:
                _, dx, dy = best
                self.move(l, dx, dy)

    def fix_aspect(self) -> None:
        for l in self.lives:
            if l.unit.is_group:
                continue   # uniform scaling of the group keeps member aspects
            for eid, b in list(l.boxes.items()):
                el = self.by_id[eid]
                if el["type"] != "image" or el.get("fit", "contain") == "cover":
                    continue
                sb = l.unit.members.get(eid)
                if sb is None or sb.w <= 0 or sb.h <= 0 or b.w <= 0:
                    continue
                aspect = sb.w / sb.h
                want_h = b.w / aspect
                if abs(want_h - b.h) > 0.5:
                    l.boxes[eid] = Box(b.x, b.cy - want_h / 2.0, b.w, want_h)
                    self.invalidate(l)

    def fix_overlaps(self) -> None:
        n = len(self.lives)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.lives[i], self.lives[j]
                pa, pb = self.parts(a), self.parts(b)
                # Overlapped by design: only a clearly bigger overlap is a
                # defect; unrelated pairs must also keep the minimum gap.
                if self.rules.conflict(a, b, pa, pb, gap=self.gap):
                    self._separate(a, b)

    def _separate(self, a: Live, b: Live) -> None:
        va, vb = self.vis(a), self.vis(b)
        g = self.gap
        # Candidate moves: (unit, dx, dy). b is later in reading order.
        moves = [
            (b, 0.0, va.bottom + g - vb.y),        # b down
            (b, va.right + g - vb.x, 0.0),         # b right
            (b, 0.0, -(vb.bottom + g - va.y)),     # b up
            (b, -(vb.right + g - va.x), 0.0),      # b left
            (a, 0.0, -(va.bottom + g - vb.y)),     # a up
            (a, -(va.right + g - vb.x), 0.0),      # a left
            (a, 0.0, vb.bottom + g - va.y),        # a down
            (a, vb.right + g - va.x, 0.0),         # a right
        ]
        sa, sb = a.unit.box, b.unit.box
        vertical_src = sa.bottom <= sb.y + 1.0 or sb.bottom <= sa.y + 1.0
        horizontal_src = sa.right <= sb.x + 1.0 or sb.right <= sa.x + 1.0
        best = None
        for l, dx, dy in moves:
            dist = abs(dx) + abs(dy)
            if dist < EPS:
                continue
            cost = dist
            if dy and vertical_src and not horizontal_src:
                cost *= 0.6          # keep the source's above/below relation
            if dx and horizontal_src and not vertical_src:
                cost *= 0.6
            if l is a:
                cost *= 1.25         # prefer moving the later / less important piece
            if l.importance > (b if l is a else a).importance + 1e-9:
                cost *= 1.5
            nb = self.vis(l)
            nb = Box(nb.x + dx, nb.y + dy, nb.w, nb.h)
            full = self.frame(l)
            full = Box(full.x + dx, full.y + dy, full.w, full.h)
            if not self.on_canvas(full):
                continue
            if self.collides(l, nb, ignore=[(b if l is a else a).unit.key]):
                continue
            if best is None or cost < best[0]:
                best = (cost, l, dx, dy)
        if best is not None:
            _, l, dx, dy = best
            self.move(l, dx, dy)
            return
        # Nothing can move: shrink the less important (or later) unit away from
        # the other, but never below MIN_SHRINK of its materialized size in
        # total (one pass per round could otherwise shrink it to nothing).
        victim, other = (b, a)
        if a.importance < b.importance - 1e-9:
            victim, other = (a, b)
        if victim.unit.key in self.protect and other.unit.key not in self.protect:
            victim, other = other, victim      # v9: a text the critic raised to its floor does not give way
        vv, vo = self.vis(victim), self.vis(other)
        options = []
        # vertical shrink: anchor at the far edge from `other`
        if vv.cy >= vo.cy:
            need = vo.bottom + g - vv.y
            options.append((1.0 - need / max(vv.h, 1.0), victim.box.x, victim.box.bottom, "v"))
        else:
            need = vv.bottom + g - vo.y
            options.append((1.0 - need / max(vv.h, 1.0), victim.box.x, victim.box.y, "v"))
        if vv.cx >= vo.cx:
            need = vo.right + g - vv.x
            options.append((1.0 - need / max(vv.w, 1.0), victim.box.right, victim.box.y, "h"))
        else:
            need = vv.right + g - vo.x
            options.append((1.0 - need / max(vv.w, 1.0), victim.box.x, victim.box.y, "h"))
        options = [o for o in options if o[0] * victim.shrunk >= MIN_SHRINK]
        if not options:
            return
        f, ax, ay, _ = max(options, key=lambda o: o[0])
        # Anchor so the far edge stays put: scaling about (ax, ay) keeps that corner fixed.
        fb = victim.box
        ax = fb.right if ax >= fb.right - EPS else fb.x
        ay = fb.bottom if ay >= fb.bottom - EPS else fb.y
        f = min(f, 0.98)
        victim.shrunk *= f
        self.scale(victim, f, ax, ay)

    def fix_on_canvas(self) -> None:
        t, mg = self.target, self.margin
        aw, ah = t.w - 2 * mg, t.h - 2 * mg
        for l in self.lives:
            fb = self.frame(l)
            if fb.w > aw + EPS or fb.h > ah + EPS:
                f = min(aw / max(fb.w, 1e-6), ah / max(fb.h, 1e-6))
                self.scale(l, f, fb.cx, fb.cy)
                fb = self.frame(l)
            dx = dy = 0.0
            if fb.x < mg:
                dx = mg - fb.x
            elif fb.right > t.w - mg:
                dx = t.w - mg - fb.right
            if fb.y < mg:
                dy = mg - fb.y
            elif fb.bottom > t.h - mg:
                dy = t.h - mg - fb.bottom
            if abs(dx) > EPS or abs(dy) > EPS:
                self.move(l, dx, dy)

    def snap(self) -> None:
        tol = self.snap_tol
        order = sorted(self.lives, key=lambda l: (-l.importance, l.unit.reading_index))
        for idx, l in enumerate(order):
            refs = order[:idx]
            if not refs:
                continue
            fb = self.frame(l)
            for axis in ("x", "y"):
                best = None
                for r in refs:
                    rb = self.frame(r)
                    if axis == "x":
                        pairs = ((fb.cx, rb.cx), (fb.x, rb.x), (fb.right, rb.right))
                    else:
                        pairs = ((fb.cy, rb.cy), (fb.y, rb.y), (fb.bottom, rb.bottom))
                    for mine, theirs in pairs:
                        d = theirs - mine
                        if EPS < abs(d) <= tol and (best is None or abs(d) < abs(best)):
                            best = d
                if best is None:
                    continue
                dx, dy = (best, 0.0) if axis == "x" else (0.0, best)
                nb = self.vis(l)
                nb = Box(nb.x + dx, nb.y + dy, nb.w, nb.h)
                full = Box(fb.x + dx, fb.y + dy, fb.w, fb.h)
                if self.on_canvas(full) and not self.collides(l, nb):
                    self.move(l, dx, dy)
                    fb = self.frame(l)

    def even_gaps(self) -> None:
        for axis in ("y", "x"):
            for chain in self._chains(axis):
                if len(chain) < 3:
                    continue
                boxes = [self.vis(l) for l in chain]
                if axis == "y":
                    gaps = [boxes[i + 1].y - boxes[i].bottom for i in range(len(boxes) - 1)]
                else:
                    gaps = [boxes[i + 1].x - boxes[i].right for i in range(len(boxes) - 1)]
                lo, hi = min(gaps), max(gaps)
                if lo < 0.0 or hi - lo <= self.snap_tol or hi > EVEN_GAP_RATIO * max(lo, 1.0):
                    continue
                mean = sum(gaps) / len(gaps)
                # Shift the intermediates so every gap equals the mean (first and last stay).
                pos = boxes[0].bottom if axis == "y" else boxes[0].right
                planned = []
                ok = True
                for l, vb in zip(chain[1:-1], boxes[1:-1]):
                    want = pos + mean
                    d = want - (vb.y if axis == "y" else vb.x)
                    nb = Box(vb.x + (d if axis == "x" else 0.0), vb.y + (d if axis == "y" else 0.0), vb.w, vb.h)
                    full = self.frame(l)
                    full = Box(full.x + (d if axis == "x" else 0.0), full.y + (d if axis == "y" else 0.0), full.w, full.h)
                    if not self.on_canvas(full) or self.collides(l, nb, ignore=[c.unit.key for c in chain]):
                        ok = False
                        break
                    planned.append((l, d))
                    pos = nb.bottom if axis == "y" else nb.right
                if not ok:
                    continue
                for l, d in planned:
                    if axis == "y":
                        self.move(l, 0.0, d)
                    else:
                        self.move(l, d, 0.0)

    def _chains(self, axis: str) -> list[list[Live]]:
        """Runs of units that follow each other along `axis` with substantial
        overlap on the other axis (a stack for "y", a row for "x")."""
        lives = sorted(self.lives, key=lambda l: self.vis(l).y if axis == "y" else self.vis(l).x)
        nxt: dict[str, Live] = {}
        prev: set[str] = set()
        for i, a in enumerate(lives):
            va = self.vis(a)
            best = None
            for b in lives[i + 1:]:
                vb = self.vis(b)
                if axis == "y":
                    gap = vb.y - va.bottom
                    ov = min(va.right, vb.right) - max(va.x, vb.x)
                    span = min(va.w, vb.w)
                else:
                    gap = vb.x - va.right
                    ov = min(va.bottom, vb.bottom) - max(va.y, vb.y)
                    span = min(va.h, vb.h)
                if gap < -0.5 or ov < min(1.0, 0.5 * span):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, b)
            if best is not None and best[1].unit.key not in prev:
                nxt[a.unit.key] = best[1]
                prev.add(best[1].unit.key)
        chains = []
        for a in lives:
            if a.unit.key in prev:
                continue
            chain = [a]
            while chain[-1].unit.key in nxt:
                chain.append(nxt[chain[-1].unit.key])
            if len(chain) >= 2:
                chains.append(chain)
        return chains

    # -- output -------------------------------------------------------------------

    def write_back(self) -> None:
        for l in self.lives:
            for eid, b in l.boxes.items():
                el = self.by_id[eid]
                el["x"], el["y"], el["width"], el["height"] = b.x, b.y, b.w, b.h
                if eid in l.fonts:
                    apply_font_size(el, l.fonts[eid])


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _is_straddle(tc) -> bool:
    return is_straddle(getattr(tc, "slices", None) or [])


def is_straddle(slices: list[float]) -> bool:
    """One slice of a text reads badly (< STRADDLE_MIN) while another reads
    well (>= STRADDLE_MAX) and STRADDLE_RATIO times better: the text lies
    across a sharp edge (a photo's border, a panel's side)."""
    if not slices:
        return False
    lo, hi = min(slices), max(slices)
    return lo < STRADDLE_MIN and hi >= STRADDLE_MAX and hi >= STRADDLE_RATIO * lo


def overlap_fraction(a: Box, b: Box) -> float:
    """Shared area as a fraction of the smaller box (0 = disjoint)."""
    small = min(a.area, b.area)
    if small <= 0:
        return 0.0
    return a.intersection_area(b) / small


def source_overlap_fractions(analysis: Analysis, measurer: TextMeasurer | None = None) -> dict[tuple[str, str], float]:
    """Overlap fraction of every pair of content units in the *source*
    (visible boxes: painted text, rotated extents), cached on the analysis.
    What the designer overlapped on purpose is not a defect when kept."""
    m = measurer or default_measurer()
    if analysis.source_visible is None:
        by_id = analysis.source_elements
        vis: dict[str, list[Box]] = {}
        for u in analysis.units:
            if not u.is_content:
                continue
            boxes = {eid: u.members[eid] for eid in u.ids if eid in by_id}
            if not boxes:
                continue
            live = Live(u, boxes, {eid: float(by_id[eid]["font_size"]) for eid in boxes if by_id[eid]["type"] == "text"})
            try:
                vis[u.key] = unit_visible_parts(live, by_id, m)
            except Exception:  # pragma: no cover - odd source element
                vis[u.key] = [u.box]
        analysis.source_visible = vis   # type: ignore[assignment]
    vis = analysis.source_visible
    out: dict[tuple[str, str], float] = {}
    keys = list(vis)
    units = {u.key: u for u in analysis.units}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            f = parts_overlap_fraction(vis[a], vis[b])
            if f > 0.0 and not text_graze(units[a], units[b], analysis.source_elements, m):
                out[_pair_key(a, b)] = f
    return out


def text_graze(a: Unit, b: Unit, by_id: dict[str, dict[str, Any]], m: TextMeasurer,
               em: float = GRAZE_EM) -> bool:
    """Is the source overlap of these two units only a sliver of text on
    text? Two unrotated texts whose painted boxes intersect by less than
    `em` x the smaller font (in either direction) ran into each other: the
    last line of one over the first line of the next in a stack, a long
    word grazing its neighbour (fonts substituted, boxes resized by a
    template converter). A designed lockup (a script word over a caps
    word, a number over its label) overlaps by a full line or more. A
    graze is not a designed overlap: the output must not keep it, so the
    source-overlap table leaves such pairs out and the rectifier pushes
    them apart."""
    ta = [i for i in a.ids if i in by_id and by_id[i]["type"] == "text"]
    tb = [i for i in b.ids if i in by_id and by_id[i]["type"] == "text"]
    if not ta or not tb:
        return False
    thin_seen = False
    for ia in a.ids:
        if ia not in by_id:
            continue
        ea = by_id[ia]
        va = visible_box(ea, a.members[ia], None, m)
        for ib in b.ids:
            if ib not in by_id:
                continue
            eb = by_id[ib]
            vb = visible_box(eb, b.members[ib], None, m)
            if not va.overlaps(vb, tol=0.0):
                continue
            if ea["type"] != "text" or eb["type"] != "text":
                return False          # a shape or image is involved: not a text graze
            if float(ea.get("rotation", 0.0) or 0.0) or float(eb.get("rotation", 0.0) or 0.0):
                return False
            fa, fb = float(ea["font_size"]), float(eb["font_size"])
            la, lb = ink_lines(ea, a.members[ia], None, m), ink_lines(eb, b.members[ib], None, m)
            same_word = " ".join(str(ea.get("content", "")).split()) == " ".join(str(eb.get("content", "")).split())
            if max(lines_inside(lb, la), lines_inside(la, lb)) >= GRAZE_CONTAINED and (
                    same_word or max(fa, fb) >= LOCKUP_FONT_RATIO * min(fa, fb)):
                return False          # a label laid on a figure, or a shadow copy of the same word (v6): a lockup
            iw = min(va.right, vb.right) - max(va.x, vb.x)
            ih = min(va.bottom, vb.bottom) - max(va.y, vb.y)
            font = min(float(ea["font_size"]), float(eb["font_size"]))
            if min(iw, ih) >= em * font:
                return False          # a full line or more: a lockup
            thin_seen = True
    return thin_seen


# ---------------------------------------------------------------------------
# which overlaps are allowed (v3 source-relative rule + v6 layering)


def _on_fraction(text_parts: list[Box], other_parts: list[Box]) -> float:
    """How much of a text's ink lies on the other unit (largest part pair)."""
    return max((o.contains_fraction(t) for t in text_parts for o in other_parts if t.area > 0), default=0.0)


class OverlapRules:
    """Decides whether two content units may overlap as they do in a layout.
    Built once per source (cached on the analysis) and shared by the
    rectifier, the scorer, the eval metrics and the tests.

    A pair may overlap as much as it did in the source plus OVERLAP_TOL
    (v3). Beyond that (v6, from the designer references):

      * text on text: never (a lockup may keep its overlap, not grow it);
      * a pair the designer layered where one side is a picture or a shape
        (a sticker on the photo, a figure in front of a wall) may overlap up
        to LAYER_GROW x the source amount, since a non-uniform canvas
        changes how much of the smaller piece lands on the bigger one;
      * a text at least HOSTED_INSIDE inside a shape / image drawn *below*
        it is hosted by it (a label on a block it was moved onto): never a
        collision, the contrast and straddle rules judge its readability;
      * a text whose role the designer layered over a picture / shape in
        the source (ink >= LAYER_SRC_MIN on it) may be layered over a
        picture / shape again when at least LAYER_MIN of its ink is on it
        (a headline that sat on the hero may sit on the hero in every
        format, not only on the identical pair it touched before).
    """

    def __init__(self, analysis: Analysis, m: TextMeasurer):
        self.src_frac = source_overlap_fractions(analysis, m)
        by_id = analysis.source_elements
        order = {eid: i for i, eid in enumerate(by_id)}
        z = {eid: float(el["z_index"]) if el.get("z_index") is not None else float(order[eid])
             for eid, el in by_id.items()}
        self.has_text: dict[str, bool] = {}
        self.z_top: dict[str, float] = {}
        self.z_bottom: dict[str, float] = {}
        self.role: dict[str, str] = {}
        for u in analysis.units:
            ids = [i for i in u.ids if i in by_id]
            self.has_text[u.key] = any(by_id[i]["type"] == "text" for i in ids)
            self.z_top[u.key] = max((z[i] for i in ids), default=0.0)
            self.z_bottom[u.key] = min((z[i] for i in ids), default=0.0)
            self.role[u.key] = u.role
        # Roles the designer layered over a picture / shape in the source.
        self.layered_roles: set[str] = set()
        vis = analysis.source_visible or {}
        for (a, b), f in self.src_frac.items():
            ta, tb = self.has_text.get(a, False), self.has_text.get(b, False)
            if ta == tb:
                continue
            text, other = (a, b) if ta else (b, a)
            if _on_fraction(vis.get(text, []), vis.get(other, [])) >= LAYER_SRC_MIN:
                self.layered_roles.add(self.role[text])

    def conflict(self, a: Live, b: Live, pa: list[Box], pb: list[Box], gap: float = 0.0,
                 tol: float = OVERLAP_TOL) -> bool:
        """Do these two units (visible parts `pa` / `pb`) overlap in a way the
        source does not excuse? With `gap` > 0 (the rectifier) two unrelated
        units closer than the gap also conflict."""
        ka, kb = a.unit.key, b.unit.key
        src = self.src_frac.get(_pair_key(ka, kb), 0.0)
        if not parts_overlap(pa, pb, tol=1.0):
            return gap > 0.0 and src <= 0.0 and parts_overlap(pa, pb, tol=-gap)
        f = parts_overlap_fraction(pa, pb)
        if f <= src + tol:
            return False
        ta, tb = self.has_text.get(ka, False), self.has_text.get(kb, False)
        if ta and tb:
            return True
        if src > 0.0 and f <= LAYER_GROW * src:
            return False
        if ta != tb:
            text, other = (a, b) if ta else (b, a)
            pt, po = (pa, pb) if ta else (pb, pa)
            on = _on_fraction(pt, po)
            if on >= HOSTED_INSIDE and self.z_bottom[text.unit.key] > self.z_top[other.unit.key]:
                return False
            if on >= LAYER_MIN and self.role[text.unit.key] in self.layered_roles:
                return False
        return True


def overlap_rules(analysis: Analysis, measurer: TextMeasurer | None = None) -> OverlapRules:
    """The source's overlap rules, built once and cached on the analysis."""
    rules = getattr(analysis, "_overlap_rules", None)
    if rules is None:
        rules = OverlapRules(analysis, measurer or default_measurer())
        analysis._overlap_rules = rules   # type: ignore[attr-defined]
    return rules


def unwanted_overlaps(analysis: Analysis, elements: list[dict[str, Any]],
                      measurer: TextMeasurer | None = None, tol: float = OVERLAP_TOL) -> list[tuple[str, str, float]]:
    """Pairs of content units whose visible boxes overlap in `elements` by
    more than they did in the source (by more than `tol` of the smaller box,
    so a graze is not a collision and a sticker that sat on the photo may
    still sit on it), with the overlap area. Shared by the scorer, the
    rectifier, the eval metrics and the tests."""
    m = measurer or default_measurer()
    by_id = {el["id"]: el for el in elements}
    lives = _lives(analysis, by_id)
    vis = {l.unit.key: unit_visible_parts(l, by_id, m) for l in lives}
    rules = overlap_rules(analysis, m)
    out = []
    for i, a in enumerate(lives):
        for b in lives[i + 1:]:
            pa, pb = vis[a.unit.key], vis[b.unit.key]
            if rules.conflict(a, b, pa, pb, tol=tol):
                out.append((a.unit.key, b.unit.key, parts_intersection(pa, pb)))
    return out
