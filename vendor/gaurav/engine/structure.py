"""Preserve-structure candidate family (v6).

The floor plans rebuild a design from a template for the target's ratio
family. A designer resizing an *editorial* layout does something else: she
reads the source's macro structure and re-proportions it. This module does
that for the structures the designer references show:

  bands    horizontal bands in reading order: a header (logo, date block,
           headline), a band made of colour zones and photos (a red panel
           beside the venue photo), a footer (legal line). Bands keep their
           order and their relative weights; the zones keep their split.
  split    one band split into columns: a colour panel carrying the copy
           beside a photo (a banner); the panel stays full-bleed.
  lockup   one zone carrying everything (a quote laid on a photo): the zone
           scales to the canvas and the lockup stays on it.

`detect_structure` reads the structure out of the analyzer's units: *zones*
are panels and big pictures / shapes (>= ZONE_MIN_AREA of the canvas); a
content unit lying >= CARRY_MIN on a zone drawn below it is *carried* by the
zone (and travels inside it); a zone >= SUB_ZONE_INSIDE inside a bigger
zone drawn below it is a *card* carried by that zone. Zones and free units
are projected on the source's stack axis (vertical for tall / square
sources, horizontal for wide / banner ones) and merged into bands; a band
holding zones side by side (on their *visible* extents, a panel drawn over
a photo hides part of it) is a split.

`structure_relayout` lays the tree out on the target: text leaves re-wrap
with the floor plan's text machinery (`engine.floorplan._TextItem`, one
global shrink factor k found by `_largest_k`), panels are elastic (never
smaller than the content they carry plus their padding), pictures keep
their aspect and take what the band gives them, leftover room goes to the
zones in proportion to their source weights. `StructureOptions` make the
2-3 variants `engine.candidates` enumerates: keep the stack axis or flip it
(bands become columns on a banner, a banner's columns become bands on a
poster), and mirror the split order. Nothing here looks at element ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from engine.analyze import (
    Analysis,
    BACKGROUND,
    BADGE,
    CTA,
    DECORATIVE,
    FINE_PRINT,
    HEADLINE,
    HERO,
    IMAGE,
    LOGO,
    PANEL,
    SHAPE,
    Unit,
)
from engine.floorplan import (
    BANNER,
    HERO_FLOOR_FRAC,
    RAIL_W_FRAC,
    LOGO_FLOOR_FRAC,
    PRIO_HERO,
    PRIO_MINOR,
    PRIO_PRIMARY,
    PRIO_SECONDARY,
    SQUARE,
    STAMP_FLOOR_FRAC,
    TALL,
    VERTICAL_LEN_SHARE,
    WIDE,
    Metrics,
    _Builder,
    _Chips,
    _Item,
    _Scaled,
    _TextItem,
    _largest_k,
    _place_column,
    ratio_family,
    slide_backdrop,
)
from engine.geometry import Box, clamp
from engine.relayout import Placement, _cover_box, _edges_touched, _map_decorative, _place_members, decorative_boxes, stack_boxes
from engine.textfit import TextMeasurer, default_measurer

ZONE_MIN_AREA = 0.10        # a picture / shape unit covering this share of the canvas is a zone (panels always are)
DECO_ZONE_MIN_AREA = 0.25   # a bleeding ("decorative") photo covering this share of the canvas is a zone too
CARRY_MIN = 0.30            # a content unit this much on a zone drawn below it travels inside the zone
SUB_ZONE_INSIDE = 0.80      # a zone this much inside a bigger zone drawn below it is a card carried by it
BAND_OVERLAP = 0.15         # two blocks overlapping on the stack axis by this share of the smaller share one band
SPLIT_OVERLAP = 0.20        # zones overlapping on the cross axis by less than this (of the smaller) stand side by side
ZONE_FLOOR_SHARE = 0.45     # a zone band keeps at least this share of its proportional length when text needs room
ZONE_PAD_MIN = 0.02         # padding inside a panel, fraction of min(W, H), at least
TEXT_W_FRAC = 0.94          # a text column uses this share of its leaf's width
CENTER_TOL = 0.08           # a leaf whose hull centre is within this of the canvas centre is centred
UNSPLIT_RATIO = 0.4         # cross extent under this x the stack length: a split's leaves stack instead
TEXT_SHARE_STRETCH = 1.5    # a text column laid along x may wrap within this x its source share of the length ...
TEXT_MIN_SHARE = 0.2        # ... and at least this share of it


# ---------------------------------------------------------------------------
# the structure of a source


@dataclass
class Leaf:
    units: list[Unit]                 # free content units (text leaf) or the zone's carried units, reading order
    zone: Unit | None = None          # the zone this leaf is (panel / picture), None for a text leaf
    cards: list["Leaf"] = field(default_factory=list)   # zone leaf: cards (sub-zones) carried inside it
    box: Box = field(default_factory=lambda: Box(0, 0, 1, 1))   # source hull (visible extent for zones)

    @property
    def is_zone(self) -> bool:
        return self.zone is not None


@dataclass
class Band:
    leaves: list[Leaf]                # one leaf, or the split's leaves along the cross axis
    box: Box

    @property
    def is_split(self) -> bool:
        return len(self.leaves) > 1

    @property
    def has_zone(self) -> bool:
        return any(l.is_zone for l in self.leaves)


@dataclass
class Structure:
    axis: str                         # "y": bands stacked top to bottom; "x": columns left to right
    bands: list[Band]
    kind: str                         # bands | split | lockup

    @property
    def zones(self) -> list[Unit]:
        return [l.zone for b in self.bands for l in b.leaves if l.zone is not None]


@dataclass(frozen=True)
class StructureOptions:
    flip: bool = False                # lay the bands along the other axis (bands <-> columns)
    swap: bool = False                # mirror the order of a split's leaves
    spacing: float = 1.0

    @property
    def name(self) -> str:
        parts = []
        if self.flip:
            parts.append("flip")
        if self.swap:
            parts.append("swap")
        if self.spacing != 1.0:
            parts.append(f"spacing={self.spacing:g}")
        return ", ".join(parts) or "keep"


def structure_variants(structure: Structure, target: Box, src: Box) -> list[StructureOptions]:
    """The variants worth trying for this structure on this target: keep the
    axis, flip it when the target's orientation differs from the source's,
    and mirror the split when there is one."""
    out = [StructureOptions()]
    if _orientation(target) != _orientation(src):
        out.append(StructureOptions(flip=True))
    if any(b.is_split for b in structure.bands):
        out.append(StructureOptions(swap=True))
    return out[:3]


def _orientation(canvas: Box) -> str:
    return "y" if ratio_family(canvas) in (TALL, SQUARE) else "x"


def _z_order(elements: dict[str, dict[str, Any]]) -> dict[str, float]:
    order = {eid: i for i, eid in enumerate(elements)}
    return {eid: float(el["z_index"]) if el.get("z_index") is not None else float(order[eid])
            for eid, el in elements.items()}


def _has_text(u: Unit, elements: dict[str, dict[str, Any]]) -> bool:
    return any(elements[i]["type"] == "text" for i in u.ids if i in elements)


def _is_zone(u: Unit, elements: dict[str, dict[str, Any]], canvas: Box) -> bool:
    if u.role == PANEL:
        return True
    if u.role == LOGO or _has_text(u, elements):
        return False
    if u.primary is None or elements[u.primary]["type"] not in ("image", "shape"):
        return False
    if any(abs(float(elements[i].get("rotation", 0.0) or 0.0)) > 1.0 for i in u.ids if i in elements):
        return False          # a rotated block's frame box is not where it is drawn: not a zone
    on_canvas = canvas.intersection_area(u.box)
    if u.role == DECORATIVE:
        # a photo band running off the canvas is "decorative" to the analyzer
        # (it bleeds); for the structure it is a zone when it covers a lot
        return elements[u.primary]["type"] == "image" and on_canvas >= DECO_ZONE_MIN_AREA * canvas.area
    return u.is_content and on_canvas >= ZONE_MIN_AREA * canvas.area


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _merge_intervals(blocks: list[tuple[float, float, Any]], tol_share: float) -> list[list[Any]]:
    """Group blocks whose intervals overlap (transitively) by more than
    `tol_share` of the smaller interval; groups come back in axis order."""
    blocks = sorted(blocks, key=lambda b: (b[0], b[1]))
    groups: list[tuple[float, float, list[Any]]] = []
    for lo, hi, item in blocks:
        placed = False
        for g in groups:
            ov = _overlap_1d(lo, hi, g[0], g[1])
            small = max(1.0, min(hi - lo, g[1] - g[0]))
            if ov > tol_share * small:
                g[2].append(item)
                groups[groups.index(g)] = (min(g[0], lo), max(g[1], hi), g[2])
                placed = True
                break
        if not placed:
            groups.append((lo, hi, [item]))
    # a later block can bridge two earlier groups: merge until stable
    changed = True
    while changed and len(groups) > 1:
        changed = False
        groups.sort(key=lambda g: g[0])
        for i in range(len(groups) - 1):
            a, b = groups[i], groups[i + 1]
            ov = _overlap_1d(a[0], a[1], b[0], b[1])
            small = max(1.0, min(a[1] - a[0], b[1] - b[0]))
            if ov > tol_share * small:
                groups[i] = (min(a[0], b[0]), max(a[1], b[1]), a[2] + b[2])
                del groups[i + 1]
                changed = True
                break
    groups.sort(key=lambda g: g[0])
    return [g[2] for g in groups]


def _clip(box: Box, canvas: Box) -> Box:
    x0, y0 = max(box.x, canvas.x), max(box.y, canvas.y)
    x1, y1 = min(box.right, canvas.right), min(box.bottom, canvas.bottom)
    return Box(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))


def _visible_zone_box(zone: Unit, zones: list[Unit], z: dict[str, float], axis: str, canvas: Box) -> Box:
    """A zone's on-canvas extent minus the part an opaque zone drawn above
    it covers along `axis` (a panel laid over the left part of a photo
    hides it)."""
    box = _clip(zone.box, canvas)
    top = z[zone.primary] if zone.primary in z else max(z[i] for i in zone.ids)
    for other in zones:
        if other is zone or min(z[i] for i in other.ids) < top:
            continue
        ob = _clip(other.box, canvas)
        if not ob.overlaps(box, tol=1.0):
            continue
        if axis == "x":
            if ob.y <= box.y + 1 and ob.bottom >= box.bottom - 1:      # covers the full height: cut a side
                if ob.x <= box.x + 1 and ob.right < box.right:
                    box = Box(ob.right, box.y, box.right - ob.right, box.h)
                elif ob.right >= box.right - 1 and ob.x > box.x:
                    box = Box(box.x, box.y, ob.x - box.x, box.h)
        else:
            if ob.x <= box.x + 1 and ob.right >= box.right - 1:
                if ob.y <= box.y + 1 and ob.bottom < box.bottom:
                    box = Box(box.x, ob.bottom, box.w, box.bottom - ob.bottom)
                elif ob.bottom >= box.bottom - 1 and ob.y > box.y:
                    box = Box(box.x, box.y, box.w, ob.y - box.y)
    return box


def detect_structure(analysis: Analysis, elements: dict[str, dict[str, Any]] | None = None) -> Structure | None:
    """The source's macro structure, or None when it has no zone to build
    one from (a logo, a headline and a button on a gradient: the floor
    plans' territory)."""
    elements = elements or analysis.source_elements
    canvas = analysis.canvas
    z = _z_order(elements)
    units = [u for u in analysis.units if u.role != BACKGROUND and all(i in elements for i in u.ids)]
    zones = [u for u in units if _is_zone(u, elements, canvas)]
    units = [u for u in units if u.role != DECORATIVE or u in zones]
    if not zones:
        return None
    zbot = {u.key: min(z[i] for i in u.ids) for u in units}
    # a zone's own layer: the picture / panel itself, not the lines attached to it
    ztop = {u.key: (z[u.primary] if u in zones and u.primary in z else max(z[i] for i in u.ids)) for u in units}

    def carrier(u: Unit, candidates: list[Unit], min_frac: float) -> Unit | None:
        best, best_f = None, 0.0
        for zn in candidates:
            if zn is u or ztop[zn.key] >= zbot[u.key]:
                continue
            f = zn.box.contains_fraction(u.box)
            if f >= min_frac and (f > best_f + 1e-6 or (abs(f - best_f) <= 1e-6 and best is not None
                                                           and zn.box.area < best.box.area)):
                best, best_f = zn, f
        return best

    # Cards: zones inside a bigger zone drawn below them.
    parent: dict[str, Unit] = {}
    for zn in sorted(zones, key=lambda u: u.box.area):
        p = carrier(zn, [o for o in zones if o.box.area > zn.box.area], SUB_ZONE_INSIDE)
        if p is not None:
            parent[zn.key] = p
    roots = [zn for zn in zones if zn.key not in parent]
    # Carried content: the smallest zone (card before panel) that holds it.
    carried: dict[str, list[Unit]] = {zn.key: [] for zn in zones}
    free: list[Unit] = []
    for u in sorted(units, key=lambda u: u.reading_index):
        if u in zones:
            continue
        c = carrier(u, zones, CARRY_MIN)
        if c is not None:
            carried[c.key].append(u)
        else:
            free.append(u)

    def leaf_for(zn: Unit) -> Leaf:
        cards = [leaf_for(c) for c in zones if parent.get(c.key) is zn]
        return Leaf(units=carried[zn.key], zone=zn, cards=cards, box=vis.get(zn.key, _clip(zn.box, canvas)))

    axis = _orientation(canvas)
    along = (lambda b: (b.y, b.bottom)) if axis == "y" else (lambda b: (b.x, b.right))
    across = (lambda b: (b.x, b.right)) if axis == "y" else (lambda b: (b.y, b.bottom))
    # visible extents along the stack axis: a panel drawn over the left part of
    # a photo hides it, and what the viewer sees stands side by side
    vis = {zn.key: _visible_zone_box(zn, roots, z, axis, canvas) for zn in roots}
    blocks: list[tuple[float, float, Any]] = []
    for zn in roots:
        lo, hi = along(vis[zn.key])
        blocks.append((lo, hi, zn))
    for u in free:
        lo, hi = along(_clip(u.box, canvas))
        blocks.append((lo, hi, u))
    bands: list[Band] = []
    for group in _merge_intervals(blocks, BAND_OVERLAP):
        g_zones = [b for b in group if b in roots]
        g_free = [b for b in group if b not in roots]
        if not g_zones:
            bands.append(Band([Leaf(units=sorted(g_free, key=lambda u: u.reading_index),
                                    box=Box.union(u.box for u in g_free))], Box.union(u.box for u in g_free)))
            continue
        # Side by side on the cross axis? Zones first, then free units slot
        # between them by their cross-axis centre.
        cols: list[list[Any]] = []
        zblocks = []
        for zn in g_zones:
            lo, hi = across(vis[zn.key])
            zblocks.append((lo, hi, zn))
        zgroups = _merge_intervals(zblocks, SPLIT_OVERLAP)
        leaves: list[Leaf] = []
        for zg in zgroups:
            if len(zg) == 1:
                leaves.append(leaf_for(zg[0]))
            else:
                # zones piled on the cross axis (two photos one above the other
                # in a column): the biggest carries the band's leaf, the others
                # become cards of it so the stack is kept
                big = max(zg, key=lambda u: u.box.area)
                lf = leaf_for(big)
                lf.cards += [leaf_for(o) for o in zg if o is not big]
                leaves.append(lf)
        leaves.sort(key=lambda l: across(vis[l.zone.key])[0])
        # free units of the band: a text leaf before / between / after the zones
        if g_free:
            cuts = [sum(across(vis[l.zone.key])) / 2.0 for l in leaves]
            bins: dict[int, list[Unit]] = {}
            for u in g_free:
                c = sum(across(u.box)) / 2.0
                k = sum(1 for cut in cuts if c > cut)
                bins.setdefault(k, []).append(u)
            merged: list[Leaf] = []
            for k in range(len(leaves) + 1):
                if k in bins:
                    us = sorted(bins[k], key=lambda u: u.reading_index)
                    merged.append(Leaf(units=us, box=Box.union(u.box for u in us)))
                if k < len(leaves):
                    merged.append(leaves[k])
            leaves = merged
        bands.append(Band(leaves, Box.union(vis[b.key] if b in roots else b.box for b in group)))
    if len(bands) == 1 and len(bands[0].leaves) == 1 and bands[0].leaves[0].is_zone:
        kind = "lockup"
    elif len(bands) == 1:
        kind = "split"
    else:
        kind = "bands"
    return Structure(axis=axis, bands=bands, kind=kind)


# ---------------------------------------------------------------------------
# layout items


class _Col(_Item):
    """A vertical column of items (texts, buttons, badges, cards) with the
    floor plan's gap policy; `align` is left / center / right."""

    def __init__(self, items: list[_Item], gaps: list[float], align: str, met: Metrics):
        self.items = items
        self.gaps = gaps
        self.align = align
        self.met = met
        self.prio = PRIO_PRIMARY
        self.tier = min((it.tier for it in items), default=0)

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        w = h = 0.0
        for it in self.items:
            iw, ih = it.size(k, aw if it.applies_w_frac else aw * it.w_frac, ah)
            w, h = max(w, iw), h + ih
        self.w, self.h = w, h + sum(self.gaps)
        return self.w, self.h

    def relax(self, factor: float, tier: int = 2) -> None:
        for it in self.items:
            it.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        if not self.items:
            return
        _place_column(self.items, self.gaps, x, y, w, h, self.align, self.met, out, False, False)


class _RowOf(_Item):
    """Columns side by side (a header with a date block left and the title
    right); their widths follow the source shares, tops align."""

    def __init__(self, cols: list[_Col], shares: list[float], gap: float):
        self.cols = cols
        self.shares = shares
        self.gap = gap
        self.prio = PRIO_PRIMARY
        self.tier = min(c.tier for c in cols)

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        inner = max(1.0, aw - self.gap * (len(self.cols) - 1))
        w = h = 0.0
        for c, s in zip(self.cols, self.shares):
            cw, ch = c.size(k, inner * s, ah)
            w, h = w + cw, max(h, ch)
        self.w, self.h = w + self.gap * (len(self.cols) - 1), h
        return self.w, self.h

    def relax(self, factor: float, tier: int = 2) -> None:
        for c in self.cols:
            c.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        inner = max(1.0, w - self.gap * (len(self.cols) - 1))
        for c, s in zip(self.cols, self.shares):
            cw = inner * s
            c.place(x, y, cw, c.h, out)
            x += cw + self.gap


class _Zone(_Item):
    """A colour panel or a picture with the content it carries laid out in a
    column inside it (plus cards: nested panels). A panel is elastic and
    takes the whole slot it is given; a `contain` picture keeps its aspect
    and takes the largest box that fits the slot; a `cover` picture fills
    the slot like a panel."""

    def __init__(self, leaf: Leaf, content: _Col | None, cards: list["_Zone"], elements: dict[str, dict[str, Any]],
                 analysis: Analysis, met: Metrics, pad: tuple[float, float, float, float], floor_len: float,
                 rails: list[tuple[_TextItem, str, str]] | None = None):
        self.leaf = leaf
        self.unit = leaf.zone
        self.content = content
        self.cards = cards
        # vertical texts that ran along the zone's sides: (item, side, vpos)
        self.rails = rails or []
        self.rail_w = 0.0
        self.elements = elements
        self.analysis = analysis
        self.met = met
        self.pad = pad                      # left, top, right, bottom padding inside the zone
        self.floor_len = floor_len          # smallest extent along the stack axis (set by the band)
        self.prio = PRIO_HERO
        self.tier = 1
        el = elements[self.unit.primary or self.unit.ids[0]]
        self.is_image = el["type"] == "image" and el.get("fit", "contain") != "cover"
        if self.is_image:
            _IMAGE_ZONES.add(self.unit.key)
        else:
            _IMAGE_ZONES.discard(self.unit.key)
        self.aspect = max(self.unit.box.w, 1.0) / max(self.unit.box.h, 1.0)
        self.touch = _edges_touched(self.unit.box, analysis.canvas)
        self.inner: list[_Item] = ([content] if content is not None else []) + list(cards)
        self.gap = met.gap

    def need(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        """Smallest (w, h) the zone needs for its content at k: the rails
        along the sides (each a vertical text wrapped in its own frame at up
        to RAIL_W_FRAC of the width), the column between them."""
        l, t, r, b = self.pad
        iw, ih = max(1.0, aw - l - r), max(1.0, ah - t - b)
        rail_w = rail_h = 0.0
        for it, _side, _v in self.rails:
            rw, rh = it.size(k, iw * RAIL_W_FRAC, ih)
            rail_w += rw + self.gap
            rail_h = max(rail_h, rh)
        self.rail_w = rail_w
        cw_budget = max(1.0, iw - rail_w)
        w = h = 0.0
        for it in self.inner:
            cw, ch = it.size(k, cw_budget, ih)
            w, h = max(w, cw), h + ch
        if len(self.inner) > 1:
            h += self.gap * (len(self.inner) - 1)
        self.cw, self.ch = w, h
        if not self.inner and not self.rails:
            return 0.0, 0.0
        return w + rail_w + l + r, max(h, rail_h) + t + b

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        nw, nh = self.need(k, aw, ah)
        if self.is_image:
            w = min(aw, ah * self.aspect)
            h = w / self.aspect
            self.w, self.h = w, h
            self.fits_content = nw <= w + 0.5 and nh <= h + 0.5
        else:
            self.w, self.h = min(aw, max(nw, 1.0)), min(ah, max(nh, 1.0))
            self.fits_content = nw <= aw + 0.5 and nh <= ah + 0.5
        return self.w, self.h

    def relax(self, factor: float, tier: int = 2) -> None:
        for it in self.inner:
            it.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        if self.is_image:
            bw = min(w, h * self.aspect)
            bh = bw / self.aspect
            box = Box(x + (w - bw) / 2.0, y + (h - bh) / 2.0, bw, bh)
        else:
            box = Box(x, y, w, h)
        ps = clamp(math.sqrt(self.met.target.area / self.analysis.canvas.area), 0.25, 2.5)
        for mid, mb in _zone_member_boxes(self.unit, box, ps).items():
            out.boxes[mid] = mb
            out.font_scale[mid] = ps
        if not self.inner and not self.rails:
            return
        l, t, r, b = self.pad
        ix, iy, iw, ih = box.x + l, box.y + t, max(1.0, box.w - l - r), max(1.0, box.h - t - b)
        # rails first: along the left / right inner edge, at the height they had
        left_x, right_x = ix, ix + iw
        for it, side, vpos in self.rails:
            ry = iy if vpos == "top" else (iy + ih - it.h if vpos == "bottom" else iy + (ih - it.h) / 2.0)
            ry = max(iy, min(ry, iy + ih - it.h))
            if side == "left":
                it.place(left_x, ry, it.w, it.h, out)
                left_x += it.w + self.gap
            else:
                right_x -= it.w
                it.place(right_x, ry, it.w, it.h, out)
                right_x -= self.gap
        ix, iw = left_x, max(1.0, right_x - left_x)
        if not self.inner:
            return
        total = sum(it.h for it in self.inner) + self.gap * (len(self.inner) - 1)
        yy = iy + max(0.0, (ih - total) / 2.0)
        for it in self.inner:
            if isinstance(it, _Zone):
                cw = min(iw, max(it.w, it.cw + it.pad[0] + it.pad[2]))
                it.place(ix + (iw - cw) / 2.0, yy, cw, it.h, out)
            else:
                it.place(ix, yy, iw, it.h, out)
            yy += it.h + self.gap


_IMAGE_ZONES: set[str] = set()     # keys of zone units whose primary is a `contain` picture (set by _Zone)


def elements_is_image(unit: Unit) -> bool:
    return unit.key in _IMAGE_ZONES


def _zone_member_boxes(unit: Unit, box: Box, ps: float) -> dict[str, Box]:
    """Member boxes of a zone placed at `box`: a layered panel stack keeps
    its insets (`stack_boxes`); a picture / panel with attached shapes (the
    rules drawn through a quote on a photo) keeps the primary at `box` and
    each attachment at its offset from the primary, scaled with it."""
    if len(unit.ids) == 1 or unit.kind == "stack" or unit.primary is None:
        return stack_boxes(unit, box, ps)
    if elements_is_image(unit):
        # a photo with rules drawn through it scales as one piece: the hull is
        # aspect-fitted to `box`, every member keeps its offset inside it
        out = Placement()
        _place_members(unit, box, box.w / max(unit.box.w, 1e-6), out)
        return out.boxes
    pb = unit.members[unit.primary]
    sx, sy = box.w / max(pb.w, 1e-6), box.h / max(pb.h, 1e-6)
    su = min(sx, sy)
    out = {unit.primary: box}
    for mid, mb in unit.members.items():
        if mid == unit.primary:
            continue
        # position by the per-axis stretch (a line stays at 70 % of the photo's
        # height), size uniformly so a thin rule stays a thin rule
        cx = box.x + (mb.cx - pb.x) * sx
        cy = box.y + (mb.cy - pb.y) * sy
        w, h = mb.w * su, mb.h * su
        out[mid] = Box(cx - w / 2.0, cy - h / 2.0, max(1.0, w), max(1.0, h))
    return out


# ---------------------------------------------------------------------------
# building items from leaves


def _align_of(units: list[Unit], elements: dict[str, dict[str, Any]], within: Box) -> str:
    """How the leaf's content is aligned: the headline's text_align when it
    has one, else from where the hull sits in `within`."""
    for u in units:
        el = elements.get(u.primary or u.ids[0])
        if el is not None and el["type"] == "text" and u.role == HEADLINE:
            a = str(el.get("text_align", "left") or "left")
            if a in ("left", "center", "right"):
                return a
    hull = Box.union(u.box for u in units)
    off = (hull.cx - within.cx) / max(within.w, 1.0)
    if abs(off) <= CENTER_TOL:
        return "center"
    return "left" if off < 0 else "right"


def _item_for(u: Unit, b: _Builder, H: float, len_scale: float | None = None) -> _Item | None:
    """One layout item for a content unit, with the floor plan's floors. A
    block (logo, date sticker, button) is as big as the proportional scale
    says, but never more than its source share of the stack's length
    (`len_scale`): a 14 % date block of a poster stays 14 % of a 16:9
    canvas's height, it does not take 23 % of it from the headline."""
    el = b.elements.get(u.primary or u.ids[0])
    text_unit = el is not None and el["type"] == "text"
    if text_unit and u.role in (HEADLINE,):
        return b.text(u, PRIO_PRIMARY)
    if text_unit and u.role == CTA:
        return b.text(u, PRIO_PRIMARY, natural_width=True)
    if text_unit and u.role == FINE_PRINT:
        return b.text(u, PRIO_MINOR)
    if text_unit:
        return b.text(u, PRIO_SECONDARY)
    prop = u.box.h * (b.met.prop if len_scale is None else min(b.met.prop, len_scale))
    if u.role == LOGO:
        return b.scaled(u, prop, max(12.0, 0.035 * H), PRIO_PRIMARY, 1.0, cap_mult=1.2, floor_frac=LOGO_FLOOR_FRAC)
    if u.role == CTA:
        return b.scaled(u, prop, 0.0, PRIO_PRIMARY, 1.0, role_for_floor=CTA, cap_mult=1.15)
    if u.role == HERO:
        return b.scaled(u, prop, 0.15 * H, PRIO_HERO, 1.0, floor_frac=HERO_FLOOR_FRAC)
    if u.is_row:
        chips = []
        for part in u.parts:
            has_text = _has_text(part, b.elements)
            it = b.scaled(part, part.box.h * (b.met.prop if len_scale is None else min(b.met.prop, len_scale)),
                          6.0 if has_text else 10.0, PRIO_MINOR, 1.0,
                          role_for_floor=BADGE if has_text else None, floor_frac=STAMP_FLOOR_FRAC)
            it.anchor_unit = u
            chips.append(it)
        cl = _Chips(chips, "row", max(3.0, b.met.gap * 0.5), 1.0)
        cl.prio = b.prio(PRIO_MINOR)
        return cl
    has_text = _has_text(u, b.elements)
    return b.scaled(u, prop, 6.0 if has_text else 8.0, PRIO_MINOR if u.role in (BADGE, SHAPE, IMAGE) else PRIO_SECONDARY,
                    1.0, role_for_floor=BADGE if (has_text and u.role == BADGE) else None,
                    floor_frac=STAMP_FLOOR_FRAC if u.role in (BADGE, IMAGE) else 0.0)


def _column(units: list[Unit], b: _Builder, met: Metrics, within: Box, flat: bool,
            len_scale: float | None = None) -> _Col | None:
    """A column (or a row of columns when the units stood side by side in
    the source and `flat` is False) for `units`."""
    if not units:
        return None
    H = met.target.h
    stacks: list[list[Unit]] = []
    if not flat:
        xb = [(u.box.x, u.box.right, u) for u in units]
        stacks = _merge_intervals(xb, 0.3)
    if len(stacks) <= 1:
        return _simple_column(units, b, met, _align_of(units, b.elements, within), H, len_scale)
    cols = []
    shares = []
    for st in stacks:
        st = sorted(st, key=lambda u: u.reading_index)
        hull = Box.union(u.box for u in st)
        col = _simple_column(st, b, met, _align_of(st, b.elements, hull), H, len_scale)
        if col is not None:
            cols.append(col)
            shares.append(max(hull.w, 1.0))
    if len(cols) == 1:
        return cols[0]
    tot = sum(shares)
    row = _RowOf(cols, [s / tot for s in shares], met.gap)
    return _Col([row], [], "left", met)


def _simple_column(units: list[Unit], b: _Builder, met: Metrics, align: str, H: float,
                   len_scale: float | None = None) -> _Col | None:
    items: list[_Item] = []
    text_gap = b.text_gap()
    head_cap = None
    for u in units:
        it = _item_for(u, b, H, len_scale)
        if it is None:
            continue
        if isinstance(it, _TextItem):
            it.w_frac = TEXT_W_FRAC
            if it.vertical:
                it.len_share = VERTICAL_LEN_SHARE
            if u.role == HEADLINE and head_cap is None:
                head_cap = it.cap
        items.append(it)
    if not items:
        return None
    gaps = []
    for a, c in zip(items, items[1:]):
        if isinstance(a, _TextItem) and isinstance(c, _TextItem):
            gaps.append(text_gap(head_cap or a.cap))
        else:
            gaps.append(met.gap)
    return _Col(items, gaps, align, met)


def _zone_item(leaf: Leaf, b: _Builder, met: Metrics, analysis: Analysis, len_scale: float) -> _Zone:
    elements = b.elements
    zn = leaf.zone
    assert zn is not None
    ps = clamp(math.sqrt(met.target.area / analysis.canvas.area), 0.25, 2.5)
    pad_min = clamp(ZONE_PAD_MIN * min(met.target.w, met.target.h), 2.0, 24.0)
    inside = leaf.units + [c.zone for c in leaf.cards if c.zone is not None]
    if inside:
        hull = Box.union(u.box for u in inside)
        pads = (max(0.0, hull.x - zn.box.x) * ps, max(0.0, hull.y - zn.box.y) * ps,
                max(0.0, zn.box.right - hull.right) * ps, max(0.0, zn.box.bottom - hull.bottom) * ps)
        pads = tuple(min(max(p, pad_min), 0.25 * min(met.target.w, met.target.h)) for p in pads)
    else:
        pads = (pad_min,) * 4
    # Vertical texts that ran along the zone's sides are rails (v5's
    # `_HeroRails`): one wrapped block each, beside the column, not in it.
    rails: list[tuple[_TextItem, str, str]] = []
    column_units: list[Unit] = []
    for u in leaf.units:
        el = elements.get(u.primary or u.ids[0])
        if el is not None and el["type"] == "text" and _is_vertical_unit(u, elements) and len(leaf.units) > 1:
            it = b.text(u, PRIO_SECONDARY if u.role != HEADLINE else PRIO_PRIMARY)
            it.len_share = 1.0
            side = "left" if u.box.cx < zn.box.cx else "right"
            rel = (u.box.cy - zn.box.y) / max(zn.box.h, 1.0)
            vpos = "top" if rel < 0.35 else ("bottom" if rel > 0.65 else "center")
            rails.append((it, side, vpos))
        else:
            column_units.append(u)
    content = _column(column_units, b, met, zn.box, flat=False, len_scale=len_scale)
    cards = [_zone_item(c, b, met, analysis, len_scale) for c in leaf.cards]
    return _Zone(leaf, content, cards, elements, analysis, met, pads, 0.0, rails)


def _is_vertical_unit(u: Unit, elements: dict[str, dict[str, Any]]) -> bool:
    from engine.floorplan import is_vertical
    el = elements.get(u.primary or u.ids[0])
    return el is not None and is_vertical(float(el.get("rotation", 0.0) or 0.0))


# ---------------------------------------------------------------------------
# the layout


@dataclass
class _Slot:
    """One band's piece of the stack: its leaves as items and the source weights."""
    band: Band
    items: list[_Item]
    shares: list[float]           # cross-axis shares of the leaves (split bands)
    share: float                  # share of the stack axis in the source

    @property
    def is_zone(self) -> bool:
        return any(isinstance(it, _Zone) for it in self.items)


def structure_relayout(analysis: Analysis, target: Box, elements: dict[str, dict[str, Any]],
                       measurer: TextMeasurer | None = None, options: StructureOptions | None = None,
                       structure: Structure | None = None) -> Placement | None:
    """Lay the source's structure out on `target`; None when the source has
    no structure to keep."""
    m = measurer or default_measurer()
    opts = options or StructureOptions()
    st = structure or detect_structure(analysis, elements)
    if st is None:
        return None
    family = ratio_family(target)
    met = Metrics.for_canvas(analysis.canvas, target, family, opts.spacing, 1.0)
    src = analysis.canvas
    s = min(target.w / src.w, target.h / src.h)
    out = Placement()
    deco = decorative_boxes(analysis, target, s)
    for unit in analysis.units:
        if unit.role == BACKGROUND:
            for eid, mb in unit.members.items():
                out.boxes[eid] = _cover_box(mb, src, target)
                out.font_scale[eid] = s
        elif unit.role == DECORATIVE and unit not in st.zones:
            _place_members(unit, deco.get(unit.key) or _map_decorative(unit.box, src, target, s), s, out)
            for eid in unit.ids:
                out.font_scale[eid] = s

    b = _Builder(analysis, elements, met, m, family)
    axis = st.axis if not opts.flip else ("x" if st.axis == "y" else "y")
    L = target.h if axis == "y" else target.w         # stack length
    C = target.w if axis == "y" else target.h         # cross extent
    src_L = src.h if st.axis == "y" else src.w
    src_C = src.w if st.axis == "y" else src.h
    along = (lambda bx: (bx.y, bx.bottom)) if st.axis == "y" else (lambda bx: (bx.x, bx.right))
    across = (lambda bx: (bx.x, bx.right)) if st.axis == "y" else (lambda bx: (bx.y, bx.bottom))

    len_scale = L / src_L          # how the stack's length changed: caps the blocks' sizes
    # On a strip (the cross extent under UNSPLIT_RATIO x the stack length) a
    # split's leaves are stacked along the axis instead, in their order: a
    # red panel beside the photo becomes the panel above the photo on a
    # 160 x 600 skyscraper, as the designer does.
    unsplit = C < UNSPLIT_RATIO * L
    slots: list[_Slot] = []
    fine_slot: _Slot | None = None

    def make_item(leaf: Leaf, band: Band) -> _Item:
        if leaf.is_zone:
            return _zone_item(leaf, b, met, analysis, len_scale)
        col = _column(leaf.units, b, met, band.box if band.is_split else Box(0, 0, src_C, src_L),
                      flat=(axis != st.axis), len_scale=len_scale)
        return col if col is not None else _Col([], [], "center", met)

    for band in st.bands:
        leaves = list(reversed(band.leaves)) if opts.swap and band.is_split else band.leaves
        lo, hi = along(band.box)
        band_share = max(hi - lo, 1.0) / src_L
        if band.is_split and unsplit:
            cross_tot = sum(max(across(l.box)[1] - across(l.box)[0], 1.0) for l in leaves)
            for leaf in leaves:
                c0, c1 = across(leaf.box)
                slots.append(_Slot(band, [make_item(leaf, band)], [1.0], band_share * max(c1 - c0, 1.0) / cross_tot))
            continue
        items = [make_item(leaf, band) for leaf in leaves]
        shares = [max(across(leaf.box)[1] - across(leaf.box)[0], 1.0) for leaf in leaves]
        tot = sum(shares)
        slot = _Slot(band, items, [sh / tot for sh in shares], band_share)
        # a footer of legal copy keeps the bottom edge whatever the axis
        if (not band.is_split and not band.has_zone and band.leaves[0].units
                and all(u.role == FINE_PRINT for u in band.leaves[0].units) and fine_slot is None
                and band is st.bands[-1]):
            fine_slot = slot
        else:
            slots.append(slot)
    if not slots:
        return None
    margin = met.margin
    gap = met.gap
    zone_total = sum(sl.share for sl in slots if sl.is_zone)

    # Lengths along the stack axis at shrink factor k: text slots take their
    # natural length, zone slots at least the content they carry and a floor
    # share of their proportional length.
    def slot_cross(sl: _Slot, idx: int) -> float:
        inner = C - (0.0 if sl.is_zone else 2 * margin)
        return max(1.0, inner * sl.shares[idx] - (gap if len(sl.items) > 1 else 0.0))

    def text_budget(sl: _Slot, avail: float) -> float:
        """Length along the axis a text slot may use: along y a column is as
        tall as it needs; along x it wraps within its source share of the
        width (stretched a little), so a flipped header does not run its
        headline across the whole banner."""
        if axis == "y":
            return avail
        return clamp(TEXT_SHARE_STRETCH * sl.share * L, TEXT_MIN_SHARE * L, avail)

    def measure(sl: _Slot, k: float, avail: float) -> float:
        need = 0.0
        for i, it in enumerate(sl.items):
            cw = slot_cross(sl, i)
            if axis == "y":
                w, h = it.size(k, cw, avail)
                ln = h
            else:
                w, h = it.size(k, avail if isinstance(it, _Zone) else text_budget(sl, avail), cw)
                ln = w
            if isinstance(it, _Zone):
                if not it.fits_content:
                    ln = avail + 1.0            # content does not fit this zone at this k
                else:
                    nw, nh = it.need(k, cw if axis == "y" else avail, avail if axis == "y" else cw)
                    ln = max((nh if axis == "y" else nw), ZONE_FLOOR_SHARE * sl.share * L)
                    if it.is_image:
                        ln = min(ln if ln > 0 else 0.0, (cw / it.aspect) if axis == "y" else (cw * it.aspect))
                        ln = max(ln, min(ZONE_FLOOR_SHARE * sl.share * L, (cw / it.aspect) if axis == "y" else cw * it.aspect))
            need = max(need, ln)
        sl.length = need      # type: ignore[attr-defined]
        return need

    fine_len = 0.0

    def fits(k: float) -> bool:
        nonlocal fine_len
        total = 0.0
        avail = L - 2 * margin
        for sl in slots:
            total += measure(sl, k, avail)
        if fine_slot is not None:
            it = fine_slot.items[0]
            w, h = it.size(k, C - 2 * margin, L)
            fine_len = (h if axis == "y" else w) + gap
            total += fine_len
        total += gap * (len(slots) - 1) + 2 * margin
        return total <= L + 0.5

    all_items = [it for sl in slots for it in sl.items] + ([fine_slot.items[0]] if fine_slot else [])
    k, _ok = _largest_k(fits, all_items)

    # Leftover goes to the zones in proportion to their source weights (an
    # image zone only up to what its aspect can use), the rest to spacing.
    lengths = {id(sl): sl.length for sl in slots}   # type: ignore[attr-defined]
    used = sum(lengths.values()) + gap * (len(slots) - 1) + 2 * margin + fine_len
    leftover = max(0.0, L - used)
    if zone_total > 0 and leftover > 0:
        for _ in range(2):
            grow = {}
            for sl in slots:
                if not sl.is_zone:
                    continue
                want = leftover * sl.share / zone_total
                cap = float("inf")
                for i, it in enumerate(sl.items):
                    if isinstance(it, _Zone) and it.is_image:
                        cw = slot_cross(sl, i)
                        cap = min(cap, (cw / it.aspect if axis == "y" else cw * it.aspect) - lengths[id(sl)])
                grow[id(sl)] = max(0.0, min(want, cap))
            given = sum(grow.values())
            for sl in slots:
                lengths[id(sl)] += grow.get(id(sl), 0.0)
            leftover -= given
            if leftover <= 1.0 or given <= 1.0:
                break
    extra_gap = leftover / (len(slots) + 1) if slots else 0.0
    extra_gap = min(extra_gap, 2.0 * gap)
    leftover -= extra_gap * (len(slots) + 1)

    # Place: bands along the axis; a zone band spans the full cross extent
    # (panels full-bleed), a text band sits inside the margins.
    pos = margin + extra_gap + max(0.0, leftover) / 2.0
    for idx, sl in enumerate(slots):
        ln = lengths[id(sl)]
        inner = C - (0.0 if sl.is_zone else 2 * margin)
        cpos = 0.0 if sl.is_zone else margin
        n = len(sl.items)
        for i, it in enumerate(sl.items):
            cw = max(1.0, inner * sl.shares[i] - (gap if n > 1 else 0.0) * (n - 1) / n)
            if isinstance(it, _Zone):
                zx, zy, zw, zh = (cpos, pos, cw, ln) if axis == "y" else (pos, cpos, ln, cw)
                # a panel that touched a canvas edge in the source is flush with it again
                box = _flush(Box(zx, zy, zw, zh), it, target, idx == 0, idx == len(slots) - 1, i == 0, i == n - 1, axis)
                it.place(box.x, box.y, box.w, box.h, out)
            else:
                if axis == "y":
                    it.size(k, cw, ln)
                    it.place(cpos, pos + (ln - it.h) / 2.0, cw, it.h, out)
                else:
                    it.size(k, text_budget(sl, ln), cw)
                    it.place(pos + (ln - it.w) / 2.0, cpos + (cw - it.h) / 2.0, it.w, it.h, out)
            cpos += cw + (gap if n > 1 else 0.0)
        pos += ln + gap + extra_gap
    if fine_slot is not None:
        it = fine_slot.items[0]
        it.size(k, target.w - 2 * margin, target.h)
        it.place(margin, target.h - margin - it.h, target.w - 2 * margin, it.h, out)
    slide_backdrop(analysis, out, target, elements, m)     # v10: texts to their source backdrop, inside their slots

    # Safety net: anything the structure did not place keeps its proportions
    # and its relative position (keeps every id present).
    for unit in analysis.units:
        if all(eid in out.boxes for eid in unit.ids):
            continue
        cx, cy = unit.box.cx / src.w * target.w, unit.box.cy / src.h * target.h
        new = Box(cx - unit.box.w * s / 2, cy - unit.box.h * s / 2, unit.box.w * s, unit.box.h * s)
        _place_members(unit, new, s, out)
        for eid in unit.ids:
            out.font_scale[eid] = s
    return out


def _flush(box: Box, zone: _Zone, target: Box, first: bool, last: bool, cfirst: bool, clast: bool, axis: str) -> Box:
    """Extend a panel's slot to the canvas edges it touched in the source
    when it sits at the corresponding end of the stack / split."""
    if zone.is_image:
        return box
    left, top, right, bottom = zone.touch
    x0, y0, x1, y1 = box.x, box.y, box.right, box.bottom
    at_left = cfirst if axis == "y" else first
    at_right = clast if axis == "y" else last
    at_top = first if axis == "y" else cfirst
    at_bottom = last if axis == "y" else clast
    if left and at_left:
        x0 = 0.0
    if right and at_right:
        x1 = target.w
    if top and at_top:
        y0 = 0.0
    if bottom and at_bottom:
        y1 = target.h
    return Box(x0, y0, x1 - x0, y1 - y0)
