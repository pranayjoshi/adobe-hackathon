"""Decoration as a pattern (v10).

Decorative elements (the analyzer's role `decorative`: faint washes, shapes
bleeding off the canvas, confetti) were moved one by one, each keeping its
offset from its nearest corner (`relayout._map_decorative`). A designer
reads them as one arrangement: two circles in opposite corners, a run of
dots along the bottom edge, three blobs on a diagonal, a mirrored pair
around the centre line, a scatter. `detect_patterns` names that
arrangement from geometry alone and `place_pattern` lays it out on the
target canvas as a whole:

  corners    each item near a different corner: it stays in that corner
             with the source's inset ratios (inset / canvas size)
  edge       all items along one edge: the edge and each item's position
             along it (as a fraction of the edge) are kept
  diagonal   the centres lie on a line tilted away from the axes: the line
             keeps its direction through the centroid (canvas fractions)
             and the items their spacing ratio, scaled down until the
             pattern fits the canvas with the items' source bleed
  mirrored   items pair up as mirror images about the vertical or
             horizontal centre line: the axis is kept, each pair stays
             symmetric at its source distance (canvas fraction)
  scatter    anything else: relative positions kept in canvas fractions

Items keep today's size rule (uniform scale by the smaller axis factor).
A lone decoration is not a pattern, and neither is a bleeding photo or
wash (MAX_ITEM_AREA): those are backdrops, kept by the old corner mapping. CHAMELEON_BACKGROUND=0 or
CHAMELEON_BACKDROP_PATTERNS=0 leaves the old per-item mapping in force.
Nothing here looks at element ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from engine.geometry import Box, clamp

CORNERS = "corners"
EDGE = "edge"
DIAGONAL = "diagonal"
MIRRORED = "mirrored"
SCATTER = "scatter"
KINDS = (CORNERS, EDGE, DIAGONAL, MIRRORED, SCATTER)

CORNER_ZONE = 0.35          # an item's centre within this fraction of both edges of a corner is "in" it
EDGE_ZONE = 0.20            # ... within this fraction of one edge is "on" it
EDGE_SPREAD_MIN = 0.30      # an edge run spreads its centres over at least this share of the edge
MIRROR_TOL = 0.06           # mirrored pairs: centre mismatch (canvas fraction) and
MIRROR_SIZE = 0.6           # ... area ratio of the pair at least this
DIAG_MIN_ITEMS = 3
DIAG_FLATNESS = 0.10        # minor / major variance of the centres along the fitted line
DIAG_MIN_ANGLE = 15.0       # degrees away from both axes
DIAG_MIN_SHRINK = 0.3       # the spacing is scaled down to this share at most to fit the canvas
BLEED_TOL = 0.05            # an item may be off canvas by this share more than in the source
MAX_ITEM_AREA = 0.20        # a "decoration" covering more of the canvas than this (on-canvas area) is a bleeding
                            # photo / wash, a backdrop in its own right (engine/structure.py may treat it as a zone):
                            # not a pattern item (a circle bleeding off a corner covers far less than its box)


@dataclass
class Pattern:
    kind: str
    keys: list[str]                                   # decorative unit keys, source order
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {"kind": self.kind, "n": len(self.keys), **{k: v for k, v in self.params.items() if k != "items"}}


# ---------------------------------------------------------------------------
# detection


def _extent(box: Box, rotation: float) -> Box:
    if not rotation:
        return box
    a = math.radians(rotation)
    w = box.w * abs(math.cos(a)) + box.h * abs(math.sin(a))
    h = box.w * abs(math.sin(a)) + box.h * abs(math.cos(a))
    return Box(box.cx - w / 2.0, box.cy - h / 2.0, w, h)


def detect_patterns(units: list[Any], canvas: Box, elements: dict[str, dict[str, Any]]) -> list[Pattern]:
    """Patterns among the free decorative `units` (single-element units of
    role decorative). At most one pattern today: the arrangement of all of
    them; two or more items are needed."""
    items = []
    for u in units:
        if len(u.ids) != 1 or u.ids[0] not in elements:
            continue
        el = elements[u.ids[0]]
        box = _extent(u.box, float(el.get("rotation", 0.0) or 0.0))
        if box.area <= 0 or canvas.intersection_area(box) >= MAX_ITEM_AREA * canvas.area:
            continue
        items.append((u.key, box))
    if len(items) < 2:
        return []
    W, H = canvas.w, canvas.h
    fr = [(k, b, b.cx / W, b.cy / H) for k, b in items]
    kind, params = _classify(fr, canvas)
    params["items"] = {k: _item_params(b, canvas) for k, b, _, _ in fr}
    return [Pattern(kind, [k for k, _, _, _ in fr], params)]


def _item_params(b: Box, canvas: Box) -> dict[str, float]:
    W, H = canvas.w, canvas.h
    return {"fx": b.cx / W, "fy": b.cy / H, "w": b.w, "h": b.h,
            "off": 1.0 - canvas.contains_fraction(b),                  # share off canvas in the source
            "il": b.x / W, "it": b.y / H, "ir": (W - b.right) / W, "ib": (H - b.bottom) / H}   # insets (may be < 0)


def _classify(fr: list[tuple[str, Box, float, float]], canvas: Box) -> tuple[str, dict[str, Any]]:
    n = len(fr)
    # corners: every item in a different corner
    corners = {}
    ok = True
    for k, b, fx, fy in fr:
        cx, cy = (0.0 if fx < 0.5 else 1.0), (0.0 if fy < 0.5 else 1.0)
        if abs(fx - cx) > CORNER_ZONE or abs(fy - cy) > CORNER_ZONE or (cx, cy) in corners.values():
            ok = False
            break
        corners[k] = (cx, cy)
    if ok and n <= 4:
        return CORNERS, {"corners": corners}
    # edge run: all along one edge, spread along it
    for edge, dist, along in (("top", lambda fx, fy: fy, lambda fx, fy: fx), ("bottom", lambda fx, fy: 1.0 - fy, lambda fx, fy: fx),
                              ("left", lambda fx, fy: fx, lambda fx, fy: fy), ("right", lambda fx, fy: 1.0 - fx, lambda fx, fy: fy)):
        if all(dist(fx, fy) <= EDGE_ZONE for _, _, fx, fy in fr):
            pos = [along(fx, fy) for _, _, fx, fy in fr]
            if max(pos) - min(pos) >= EDGE_SPREAD_MIN:
                return EDGE, {"edge": edge}
    # mirrored pairs about the vertical / horizontal centre line
    for axis in ("v", "h"):
        pairs = _mirror_pairs(fr, axis)
        if pairs is not None:
            return MIRRORED, {"axis": axis, "pairs": pairs}
    # diagonal: centres on a tilted line
    if n >= DIAG_MIN_ITEMS:
        diag = _diagonal(fr, canvas)
        if diag is not None:
            return DIAGONAL, diag
    return SCATTER, {}


def _mirror_pairs(fr: list[tuple[str, Box, float, float]], axis: str) -> list[tuple[str, str]] | None:
    if len(fr) % 2:
        return None
    left = list(fr)
    pairs: list[tuple[str, str]] = []
    while left:
        k, b, fx, fy = left.pop(0)
        best = None
        for j, (k2, b2, fx2, fy2) in enumerate(left):
            if axis == "v":
                err = max(abs(fx + fx2 - 1.0), abs(fy - fy2))
                same_side = (fx < 0.5) == (fx2 < 0.5) and abs(fx - 0.5) > MIRROR_TOL
            else:
                err = max(abs(fy + fy2 - 1.0), abs(fx - fx2))
                same_side = (fy < 0.5) == (fy2 < 0.5) and abs(fy - 0.5) > MIRROR_TOL
            ratio = min(b.area, b2.area) / max(b.area, b2.area)
            if err <= MIRROR_TOL and ratio >= MIRROR_SIZE and not same_side and (best is None or err < best[0]):
                best = (err, j)
        if best is None:
            return None
        k2 = left.pop(best[1])[0]
        pairs.append((k, k2))
    return pairs


def _diagonal(fr: list[tuple[str, Box, float, float]], canvas: Box) -> dict[str, Any] | None:
    xs = [b.cx for _, b, _, _ in fr]
    ys = [b.cy for _, b, _, _ in fr]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs) / n
    syy = sum((y - my) ** 2 for y in ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = max(0.0, tr * tr / 4.0 - det)
    major, minor = tr / 2.0 + math.sqrt(disc), tr / 2.0 - math.sqrt(disc)
    if major <= 1e-6 or minor / major > DIAG_FLATNESS:
        return None
    angle = math.degrees(math.atan2(2.0 * sxy, sxx - syy) / 2.0)   # direction of the major axis
    a = abs(angle) % 180.0
    if min(a, 180.0 - a) < DIAG_MIN_ANGLE or abs(a - 90.0) < DIAG_MIN_ANGLE:
        return None
    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    along = {k: (b.cx - mx) * ux + (b.cy - my) * uy for k, b, _, _ in fr}
    across = {k: -(b.cx - mx) * uy + (b.cy - my) * ux for k, b, _, _ in fr}
    return {"angle": angle, "fcx": mx / canvas.w, "fcy": my / canvas.h, "along": along, "across": across}


# ---------------------------------------------------------------------------
# placement


def place_pattern(pattern: Pattern, src: Box, target: Box, s: float) -> dict[str, Box]:
    """Boxes (unit extents) of the pattern's items on `target`; sizes are
    the source sizes x `s`."""
    items = pattern.params["items"]
    W, H = target.w, target.h
    out: dict[str, Box] = {}
    sizes = {k: (p["w"] * s, p["h"] * s) for k, p in items.items()}
    if pattern.kind == CORNERS:
        for k, (cx, cy) in pattern.params["corners"].items():
            p, (w, h) = items[k], sizes[k]
            x = p["il"] * W if cx == 0.0 else W - p["ir"] * W - w
            y = p["it"] * H if cy == 0.0 else H - p["ib"] * H - h
            out[k] = Box(x, y, w, h)
    elif pattern.kind == EDGE:
        edge = pattern.params["edge"]
        for k, p in items.items():
            w, h = sizes[k]
            if edge in ("top", "bottom"):
                x = p["fx"] * W - w / 2.0
                y = p["it"] * H if edge == "top" else H - p["ib"] * H - h
            else:
                y = p["fy"] * H - h / 2.0
                x = p["il"] * W if edge == "left" else W - p["ir"] * W - w
            out[k] = Box(x, y, w, h)
    elif pattern.kind == MIRRORED:
        axis = pattern.params["axis"]
        for k, p in items.items():
            w, h = sizes[k]
            out[k] = Box(p["fx"] * W - w / 2.0, p["fy"] * H - h / 2.0, w, h)
        for a, b in pattern.params["pairs"]:
            # the pair stays symmetric about the centre line at a's distance
            pa, pb = items[a], items[b]
            if axis == "v":
                d = (abs(pa["fx"] - 0.5) + abs(pb["fx"] - 0.5)) / 2.0
                for k, p in ((a, pa), (b, pb)):
                    w, h = sizes[k]
                    cx = (0.5 - d if p["fx"] < 0.5 else 0.5 + d) * W
                    out[k] = Box(cx - w / 2.0, (pa["fy"] + pb["fy"]) / 2.0 * H - h / 2.0, w, h)
            else:
                d = (abs(pa["fy"] - 0.5) + abs(pb["fy"] - 0.5)) / 2.0
                for k, p in ((a, pa), (b, pb)):
                    w, h = sizes[k]
                    cy = (0.5 - d if p["fy"] < 0.5 else 0.5 + d) * H
                    out[k] = Box((pa["fx"] + pb["fx"]) / 2.0 * W - w / 2.0, cy - h / 2.0, w, h)
    elif pattern.kind == DIAGONAL:
        q = pattern.params
        ux, uy = math.cos(math.radians(q["angle"])), math.sin(math.radians(q["angle"]))
        cx, cy = q["fcx"] * W, q["fcy"] * H
        f = 1.0
        for _ in range(12):
            out = {}
            for k, p in items.items():
                w, h = sizes[k]
                t, c = q["along"][k] * s * f, q["across"][k] * s * f
                x, y = cx + t * ux - c * uy, cy + t * uy + c * ux
                out[k] = Box(x - w / 2.0, y - h / 2.0, w, h)
            if all(1.0 - target.contains_fraction(b) <= items[k]["off"] + BLEED_TOL for k, b in out.items()) or f <= DIAG_MIN_SHRINK:
                break
            f *= 0.85
    else:
        for k, p in items.items():
            w, h = sizes[k]
            out[k] = Box(p["fx"] * W - w / 2.0, p["fy"] * H - h / 2.0, w, h)
    return out


def pattern_boxes(analysis: Any, target: Box, s: float) -> dict[str, Box]:
    """Unit key -> box for every item of every pattern of `analysis` on
    `target` (empty when patterns are off or there are none)."""
    patterns = getattr(analysis, "patterns", None) or []
    if not patterns:
        return {}
    from engine.backdrop import enabled
    if not enabled("patterns"):
        return {}
    out: dict[str, Box] = {}
    for p in patterns:
        out.update(place_pattern(p, analysis.canvas, target, s))
    return out
