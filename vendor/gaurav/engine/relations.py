"""Relational graph of a design: the perceptual variables a layout is judged on.

O'Donovan, Agarwala and Hertzmann ("Learning Layouts for Single-Page Graphic
Designs", TVCG 2014) infer *alignment groups*, *grouping / proximity*,
*importance* and *reading flow* from a finished design and retarget it by
keeping those relations while an energy over alignment, balance, white
space and overlap is minimised. This module computes that graph for the
engine's units:

  alignment    which units share a left / centre / right x line or a
               top / centre / bottom y line (tolerance scaled by the canvas),
               as groups per line type and as pairs; plus the *near misses*
               (edges that almost align: the eye reads them as a mistake)
  proximity    neighbours along a stack or a row whose gap is small relative
               to their size (a headline with its subhead, a label with its
               button): the output should keep them together
  containment  a unit that sits inside another unit, or on a panel
  symmetry     units centred on a canvas axis, and mirror pairs
  importance   the analyzer's role importance per unit
  reading      reading order over the visible boxes (engine/score.py's rule)
  size         linear size share and largest font per unit (hierarchy)

`relations(analysis, elements, canvas)` is computed once for the source and
again for any output of the same source: the two graphs compare unit by
unit (same keys), so a lost alignment or a broken proximity group is a
countable, importance-weighted defect (engine/energy.py). Everything works
on *visible* boxes (painted text, rotated extents) like the scorer, and
`Relations.to_dict()` serialises the graph for debugging / the playground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from engine.analyze import Analysis, PANEL, PANEL_CARRY, Unit, _anchors
from engine.geometry import Box
from engine.rectify import Live, _lives, rotated_extent, unit_visible_box
from engine.score import reading_order
from engine.textfit import TextMeasurer, default_measurer

ALIGN_TOL_FRAC = 0.01       # two lines align when within max(ALIGN_TOL_PX, this x min(W, H))
ALIGN_TOL_PX = 2.0
NEAR_MISS_MULT = 3.0        # ... and nearly align (a visible mistake) up to this x the tolerance
PROX_SIZE = 0.6             # neighbours are proximate when their gap is at most this x the smaller extent ...
PROX_CANVAS = 0.025         # ... or this x the canvas span along the axis
OTHER_AXIS_OVERLAP = 0.5    # neighbours overlap on the other axis by this share of the smaller span
PROX_SATURATE = 0.5         # a broken proximity costs 1 when the excess gap reaches this x the canvas span
SYM_TOL = 0.02              # centred on a canvas axis when the centre is within this x the span of it
MIRROR_TOL = 0.03           # mirror pairs: mirrored box within this x the canvas span (position and size)
CONTAIN_FRAC = 0.8          # inside another unit by this share of the area

X_LINES = ("left", "cx", "right")
Y_LINES = ("top", "cy", "bottom")
LINES = X_LINES + Y_LINES


@dataclass
class Relations:
    canvas: Box
    keys: list[str]                                      # content unit keys, analyzer order
    boxes: dict[str, Box]                                # visible box per unit
    importance: dict[str, float]
    alignments: dict[str, list[list[str]]]               # line type -> groups (>= 2 units) sharing that line
    aligned_pairs: dict[tuple[str, str], tuple[str, ...]]   # pair -> line types shared
    near_misses: list[tuple[str, str, str]]              # (a, b, line type) almost aligned
    proximity: dict[tuple[str, str], str]                # pair -> axis ("x" row | "y" stack) they neighbour along
    containment: dict[str, str]                          # inner unit -> outer unit (content) or panel key
    symmetry: dict[str, tuple[bool, bool]]               # unit -> (centred on the vertical axis, on the horizontal one)
    mirror_pairs: list[tuple[str, str, str]]             # (a, b, axis) mirrored about a canvas axis
    reading: dict[str, int]                              # unit -> reading position
    anchors: dict[str, tuple[str, str]]                  # unit -> (anchor_x, anchor_y) on this canvas
    size: dict[str, float]                               # unit -> linear size share (sqrt of area / canvas area)
    fonts: dict[str, float]                              # unit -> largest font size of its texts
    tol: float = 0.0

    # -- derived views ------------------------------------------------------------

    def shares_line(self, a: str, b: str, axis: str | None = None) -> bool:
        lines = self.aligned_pairs.get(_pair(a, b), ())
        if axis == "x":
            return any(l in X_LINES for l in lines)
        if axis == "y":
            return any(l in Y_LINES for l in lines)
        return bool(lines)

    def gap(self, a: str, b: str, axis: str) -> float:
        """Empty distance between two units along `axis` (negative: overlap)."""
        ba, bb = self.boxes[a], self.boxes[b]
        if axis == "y":
            return max(bb.y - ba.bottom, ba.y - bb.bottom)
        return max(bb.x - ba.right, ba.x - bb.right)

    def to_dict(self) -> dict[str, Any]:
        def box(b: Box) -> list[float]:
            return [round(b.x, 1), round(b.y, 1), round(b.w, 1), round(b.h, 1)]

        return {
            "canvas": [self.canvas.w, self.canvas.h],
            "tol": round(self.tol, 2),
            "units": [{"key": k, "box": box(self.boxes[k]), "importance": self.importance[k],
                       "reading": self.reading.get(k), "anchors": list(self.anchors[k]),
                       "size": round(self.size[k], 4), "font": self.fonts.get(k),
                       "symmetry": list(self.symmetry[k]), "inside": self.containment.get(k)}
                      for k in self.keys],
            "alignments": {line: groups for line, groups in self.alignments.items() if groups},
            "aligned_pairs": [[a, b, list(lines)] for (a, b), lines in sorted(self.aligned_pairs.items())],
            "near_misses": [list(t) for t in self.near_misses],
            "proximity": [[a, b, axis] for (a, b), axis in sorted(self.proximity.items())],
            "mirror_pairs": [list(t) for t in self.mirror_pairs],
        }


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def align_tolerance(canvas: Box) -> float:
    return max(ALIGN_TOL_PX, ALIGN_TOL_FRAC * min(canvas.w, canvas.h))


def line_value(b: Box, line: str) -> float:
    return {"left": b.x, "cx": b.cx, "right": b.right, "top": b.y, "cy": b.cy, "bottom": b.bottom}[line]


# ---------------------------------------------------------------------------


def relations(analysis: Analysis, elements: list[dict[str, Any]], canvas: Box,
              measurer: TextMeasurer | None = None, boxes: dict[str, Box] | None = None) -> Relations:
    """The relational graph of `elements` (a materialized design of the
    analysed source, or the source itself) on `canvas`. `boxes` may hand in
    the units' visible boxes when the caller has them already."""
    m = measurer or default_measurer()
    by_id = {el["id"]: el for el in elements}
    lives = [l for l in _lives(analysis, by_id)]
    if boxes is None:
        boxes = {l.unit.key: unit_visible_box(l, by_id, m) for l in lives}
    else:
        boxes = {l.unit.key: boxes[l.unit.key] for l in lives if l.unit.key in boxes}
        lives = [l for l in lives if l.unit.key in boxes]
    keys = [l.unit.key for l in lives]
    units = {l.unit.key: l.unit for l in lives}
    tol = align_tolerance(canvas)

    importance = {k: max(float(units[k].importance), 0.05) for k in keys}
    alignments, aligned_pairs, near = _alignment_graph(keys, boxes, tol)
    proximity = _proximity(keys, boxes, canvas)
    containment = _containment(keys, boxes, analysis, by_id)
    symmetry, mirrors = _symmetry(keys, boxes, canvas)
    reading = reading_order([(k, boxes[k]) for k in keys]) if keys else {}
    anchors = {k: _anchors(boxes[k], canvas) for k in keys}
    size = {k: math.sqrt(max(boxes[k].area, 0.0) / max(canvas.area, 1.0)) for k in keys}
    fonts: dict[str, float] = {}
    for l in lives:
        fs = [float(by_id[e]["font_size"]) for e in l.unit.ids if e in by_id and by_id[e]["type"] == "text"]
        if fs:
            fonts[l.unit.key] = max(fs)
    return Relations(canvas=canvas, keys=keys, boxes=boxes, importance=importance, alignments=alignments,
                     aligned_pairs=aligned_pairs, near_misses=near, proximity=proximity, containment=containment,
                     symmetry=symmetry, mirror_pairs=mirrors, reading=reading, anchors=anchors, size=size,
                     fonts=fonts, tol=tol)


def source_relations(analysis: Analysis, measurer: TextMeasurer | None = None) -> Relations:
    """The source's graph, cached on the analysis."""
    cached = getattr(analysis, "source_relations", None)
    if cached is None:
        cached = relations(analysis, list(analysis.source_elements.values()), analysis.canvas, measurer)
        try:
            analysis.source_relations = cached   # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
    return cached


# ---------------------------------------------------------------------------


def _alignment_graph(keys: list[str], boxes: dict[str, Box], tol: float
                     ) -> tuple[dict[str, list[list[str]]], dict[tuple[str, str], tuple[str, ...]],
                                list[tuple[str, str, str]]]:
    groups: dict[str, list[list[str]]] = {}
    pairs: dict[tuple[str, str], list[str]] = {}
    near: list[tuple[str, str, str]] = []
    for line in LINES:
        vals = sorted(((line_value(boxes[k], line), k) for k in keys), key=lambda t: (t[0], t[1]))
        # single-linkage clusters along the line
        clusters: list[list[tuple[float, str]]] = []
        for v, k in vals:
            if clusters and v - clusters[-1][-1][0] <= tol:
                clusters[-1].append((v, k))
            else:
                clusters.append([(v, k)])
        groups[line] = [[k for _, k in c] for c in clusters if len(c) >= 2]
        for c in clusters:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    pairs.setdefault(_pair(c[i][1], c[j][1]), []).append(line)
        # near misses: consecutive clusters closer than NEAR_MISS_MULT x tol
        for i in range(len(clusters) - 1):
            a, b = clusters[i], clusters[i + 1]
            d = b[0][0] - a[-1][0]
            if tol < d <= NEAR_MISS_MULT * tol:
                ka, kb = a[-1][1], b[0][1]
                near.append((ka, kb, line))
    aligned = {p: tuple(ls) for p, ls in pairs.items()}
    # a pair of near-aligned edges that is aligned on another line of the same
    # axis (left edges 2 px apart but centres equal) is not a mistake
    near = [(a, b, line) for a, b, line in near
            if not any(l != line and ((l in X_LINES) == (line in X_LINES)) for l in aligned.get(_pair(a, b), ()))]
    return groups, aligned, near


def _proximity(keys: list[str], boxes: dict[str, Box], canvas: Box) -> dict[tuple[str, str], str]:
    """Nearest neighbour below / to the right of each unit, when the gap is
    small relative to the two units (or the canvas) and they overlap on the
    other axis: the pairs a viewer reads as one group."""
    out: dict[tuple[str, str], str] = {}
    for axis in ("y", "x"):
        span = canvas.h if axis == "y" else canvas.w
        order = sorted(keys, key=lambda k: (boxes[k].y, boxes[k].x) if axis == "y" else (boxes[k].x, boxes[k].y))
        for i, a in enumerate(order):
            ba = boxes[a]
            best: tuple[float, str] | None = None
            for b in order[i + 1:]:
                bb = boxes[b]
                if axis == "y":
                    gap = bb.y - ba.bottom
                    ov = min(ba.right, bb.right) - max(ba.x, bb.x)
                    other = min(ba.w, bb.w)
                    ext = min(ba.h, bb.h)
                else:
                    gap = bb.x - ba.right
                    ov = min(ba.bottom, bb.bottom) - max(ba.y, bb.y)
                    other = min(ba.h, bb.h)
                    ext = min(ba.w, bb.w)
                if gap < -0.5 or ov < min(1.0, OTHER_AXIS_OVERLAP * other):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, b)
                    if gap > max(PROX_SIZE * ext, PROX_CANVAS * span):
                        best = None
                        break   # the nearest neighbour is already too far: no proximity along this axis
            if best is not None:
                out[_pair(a, best[1])] = axis
    return out


def _containment(keys: list[str], boxes: dict[str, Box], analysis: Analysis,
                 by_id: dict[str, dict[str, Any]]) -> dict[str, str]:
    """inner -> outer: the smallest content unit that holds >= CONTAIN_FRAC of
    a unit's visible box, else the panel it sits on (>= PANEL_CARRY, the
    analyzer's rule), judged on the current geometry."""
    out: dict[str, str] = {}
    panels: list[tuple[str, Box]] = []
    for u in analysis.units:
        if u.role == PANEL:
            ids = [i for i in u.ids if i in by_id]
            if ids:
                panels.append((u.key, Box.union(rotated_extent(Box.of(by_id[i]),
                                                               float(by_id[i].get("rotation", 0.0) or 0.0))
                                                for i in ids)))
    for a in keys:
        ba = boxes[a]
        if ba.area <= 0:
            continue
        best: tuple[float, str] | None = None
        for b in keys:
            if b == a:
                continue
            bb = boxes[b]
            if bb.area <= ba.area:
                continue
            if bb.contains_fraction(ba) >= CONTAIN_FRAC and (best is None or bb.area < best[0]):
                best = (bb.area, b)
        if best is None:
            for pk, pb in panels:
                if pb.contains_fraction(ba) >= PANEL_CARRY and (best is None or pb.area < best[0]):
                    best = (pb.area, pk)
        if best is not None:
            out[a] = best[1]
    return out


def _symmetry(keys: list[str], boxes: dict[str, Box], canvas: Box
              ) -> tuple[dict[str, tuple[bool, bool]], list[tuple[str, str, str]]]:
    sym: dict[str, tuple[bool, bool]] = {}
    for k in keys:
        b = boxes[k]
        sym[k] = (abs(b.cx - canvas.w / 2.0) <= SYM_TOL * canvas.w,
                  abs(b.cy - canvas.h / 2.0) <= SYM_TOL * canvas.h)
    mirrors: list[tuple[str, str, str]] = []
    for i, a in enumerate(keys):
        ba = boxes[a]
        for b in keys[i + 1:]:
            bb = boxes[b]
            # mirrored about the vertical axis: same y / size, x reflected
            if (abs(ba.y - bb.y) <= MIRROR_TOL * canvas.h and abs(ba.w - bb.w) <= MIRROR_TOL * canvas.w
                    and abs(ba.h - bb.h) <= MIRROR_TOL * canvas.h
                    and abs((canvas.w - bb.right) - ba.x) <= MIRROR_TOL * canvas.w and abs(ba.cx - bb.cx) > 4 * MIRROR_TOL * canvas.w):
                mirrors.append((a, b, "x"))
            if (abs(ba.x - bb.x) <= MIRROR_TOL * canvas.w and abs(ba.w - bb.w) <= MIRROR_TOL * canvas.w
                    and abs(ba.h - bb.h) <= MIRROR_TOL * canvas.h
                    and abs((canvas.h - bb.bottom) - ba.y) <= MIRROR_TOL * canvas.h and abs(ba.cy - bb.cy) > 4 * MIRROR_TOL * canvas.h):
                mirrors.append((a, b, "y"))
    return sym, mirrors


# ---------------------------------------------------------------------------
# comparison helpers (used by engine/energy.py; kept here so the eval can
# print them per case)


def lost_alignments(src: Relations, out: Relations) -> list[tuple[str, str, str]]:
    """Source pairs that shared a line on an axis and share none on that
    axis any more: (a, b, axis)."""
    lost = []
    for (a, b), lines in src.aligned_pairs.items():
        if a not in out.boxes or b not in out.boxes:
            continue
        for axis, mine in (("x", X_LINES), ("y", Y_LINES)):
            if any(l in mine for l in lines) and not out.shares_line(a, b, axis):
                lost.append((a, b, axis))
    return lost


def broken_proximity(src: Relations, out: Relations) -> list[tuple[str, str, float]]:
    """Source neighbours that drifted apart: (a, b, cost) with cost in (0, 1].
    Still adjacent along the same axis but with a gap past the proximity
    threshold: the cost grows linearly with the excess gap, 1 at
    PROX_SATURATE of the canvas span (slow, so pulling a lockup back
    together always pays); no longer stacked / in a row at all (or
    overlapping now): cost 1."""
    out_pairs = []
    for (a, b), axis in src.proximity.items():
        if a not in out.boxes or b not in out.boxes:
            continue
        if _pair(a, b) in out.proximity:
            continue
        ba, bb = out.boxes[a], out.boxes[b]
        if axis == "y":
            ov = min(ba.right, bb.right) - max(ba.x, bb.x)
            other, ext, span = min(ba.w, bb.w), min(ba.h, bb.h), out.canvas.h
        else:
            ov = min(ba.bottom, bb.bottom) - max(ba.y, bb.y)
            other, ext, span = min(ba.h, bb.h), min(ba.w, bb.w), out.canvas.w
        gap = out.gap(a, b, axis)
        thresh = max(PROX_SIZE * ext, PROX_CANVAS * span)
        if ov < min(1.0, OTHER_AXIS_OVERLAP * other) or gap < -0.5:
            cost = 1.0
        else:
            cost = min(1.0, max(0.0, (gap - thresh) / (PROX_SATURATE * span)))
        if cost > 0.0:
            out_pairs.append((a, b, cost))
    return out_pairs


def lost_containment(src: Relations, out: Relations) -> list[tuple[str, str]]:
    return [(a, outer) for a, outer in src.containment.items()
            if a in out.boxes and out.containment.get(a) != outer]


def lost_symmetry(src: Relations, out: Relations) -> list[tuple[str, str]]:
    lost = []
    for k, (sx, sy) in src.symmetry.items():
        if k not in out.symmetry:
            continue
        ox, oy = out.symmetry[k]
        if sx and not ox:
            lost.append((k, "x"))
        if sy and not oy:
            lost.append((k, "y"))
    return lost


def order_discordance(src: Relations, out: Relations) -> tuple[float, list[tuple[str, str]]]:
    """Importance-weighted share of unit pairs whose reading order flipped,
    and the flipped pairs."""
    keys = [k for k in src.keys if k in out.reading and k in src.reading]
    num = den = 0.0
    flipped = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            s = src.reading[a] - src.reading[b]
            t = out.reading[a] - out.reading[b]
            if s == 0 or t == 0:
                continue
            w = min(src.importance[a], src.importance[b])
            den += w
            if s * t < 0:
                num += w
                flipped.append((a, b))
    return (num / den if den else 0.0), flipped


def unit_of(analysis: Analysis, key: str) -> Unit | None:
    for u in analysis.units:
        if u.key == key:
            return u
    return None


__all__ = ["Relations", "relations", "source_relations", "lost_alignments", "broken_proximity",
           "lost_containment", "lost_symmetry", "order_discordance", "align_tolerance", "line_value",
           "LINES", "X_LINES", "Y_LINES", "Live"]
