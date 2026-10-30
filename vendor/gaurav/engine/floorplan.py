"""Floor plans: archetype layouts for big canvas shape changes.

When the target canvas has a very different shape from the source (ratio
change above `resize.SMALL_CHANGE_MAX`), sliding things toward their
anchors leaves a tiny shrunken poster floating in the new canvas. Instead
we pick a *floor plan* by the target's ratio family and pour the content
units from `analyze()` into its slots:

  tall   (w/h <= 0.8)   column stack: logo, hero, headline, subhead, body,
                        badges row, CTA, fine print at the bottom
  square (0.8 - 1.25)   the same centred stack, slightly tighter
  wide   (1.25 - 3)     hero on one side, a column with logo, headline,
                        subhead, badges, CTA and fine print on the other
  banner (> 3)          a single row: logo | hero | text column | badges | CTA

Every slot's size is a function of one global shrink factor `k` in [0, 1]
(1 = the comfortable size for this canvas, 0 = the readable minimum). Items
shrink at different speeds (`prio`): the headline, logo, CTA and hero are
the last to give up room, subheads sooner, badges and legal copy first.
A binary search finds the largest `k` at which everything fits; whatever
room is left goes to spacing. Text is measured with the real brand fonts
(`TextMeasurer`) so the font sizes we emit really fit their boxes, images
scale uniformly, and groups move as one piece with their members' offsets
scaled (`relayout._place_members`).

Nothing here looks at element ids: roles come from the analyzer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from engine.analyze import (
    Analysis,
    BACKGROUND,
    BADGE,
    BODY,
    CTA,
    DECORATIVE,
    FINE_PRINT,
    HEADLINE,
    HERO,
    IMAGE,
    LOGO,
    SUBHEAD,
    TEXT,
    Unit,
)
from engine.geometry import Box, clamp
from engine.relayout import Placement, _cover_box, _map_decorative, _place_members, decorative_boxes, place_panels
from engine.textfit import TextMeasurer, default_measurer, scaled_letter_spacing

TALL = "tall"
SQUARE = "square"
WIDE = "wide"
BANNER = "banner"

TALL_MAX_RATIO = 0.8
SQUARE_MAX_RATIO = 1.25
WIDE_MAX_RATIO = 3.0

SHORT_CANVAS = 300.0       # px tall; below this minimum font sizes scale with the height
K_SEARCH_STEPS = 10        # k to 1/1024: fonts are integers, finer steps only cost time (v4: was 14)
VERTICAL_MIN_DEG = 60.0    # a text rotated at least this far is fitted in its own (vertical) frame

# Shrink speeds: size(k) = ideal * k ** prio (never below the item's floor).
PRIO_PRIMARY = 1.0         # headline, logo, CTA
PRIO_HERO = 1.3
PRIO_SECONDARY = 1.6       # subhead, body
PRIO_MINOR = 2.5           # badges, secondary images, fine print

# Banner headline wrapping: (max lines, smallest shrink factor k at which that
# many lines is still preferred over wrapping once more).
BANNER_LINE_POLICY = ((1, 0.75), (2, 0.45), (3, 0.0), (4, 0.0), (6, 0.0))

# Floors relative to the target canvas (gk7494's figures): fonts as a share
# of the long side, brand pieces as a share of the short side (their larger
# dimension), so a logo or a hero never vanishes on a big canvas.
FONT_FLOOR_LONG = 0.0085
FONT_FLOOR_MULT = {FINE_PRINT: 0.7, BADGE: 0.8}
LOGO_FLOOR_FRAC = 0.14
HERO_FLOOR_FRAC = 0.30
STAMP_FLOOR_FRAC = 0.08
CHIP_MAX_SHARE = 0.18       # a chip's ideal height is at most this share of the canvas height

# Tall strips (v5): on a canvas narrower than STRIP_MAX_RATIO x its height
# the fonts also have a floor relative to the *width* (a 160 px skyscraper
# is read at arm's length like any other 160 px wide thing), and a hero in
# a column takes the height the width-bound text leaves over, up to
# HERO_MAX_SHARE of the canvas height.
STRIP_MAX_RATIO = 0.5
STRIP_FONT_FRAC = {HEADLINE: 0.045, CTA: 0.035, FINE_PRINT: 0.025, BADGE: 0.022}
STRIP_FONT_FRAC_DEFAULT = 0.03
HERO_MAX_SHARE = 0.55
VERTICAL_LEN_SHARE = 0.5    # a vertical text in a column runs along at most this share of the column height
                            # before it wraps onto a second line (a spine label), and never over the whole column
RAIL_W_FRAC = 0.3           # a vertical text beside the hero (a rail) takes at most this share of the width
RAIL_LEN_SHARE = 0.6        # ... and may run past the hero up to this share of the column height to stay on one line

# Ambition (v4): the plans may size things past the source-proportional size
# when the canvas has room. The scorer credits primary units up to AMBITION x
# sqrt(target area / source area) (tailsgit's 1.2 cap); candidates with a
# bigger `scale_bias` only win when they also fit and fill the canvas.
AMBITION = 1.2
HERO_SHARES = (0.45, 0.36, 0.28)   # wide plan: hero width shares tried, biggest first
HEAD_MAX_LINES = 3                 # ... until the headline wraps on at most this many lines
HERO_SHARE_K_TOL = 0.9             # ... as long as the column keeps this share of the best k found
BANNER_COLUMN_MIN_H = 160.0        # a banner at least this tall may also use the hero-beside-column plan

# Cohesion (v5): leftover width in a banner row is not spread evenly between
# the items (two units ended up at the far edges with 80 % empty strip
# between them); each gap grows to at most max(MAX_GAP_MULT x the base gap,
# MAX_GAP_FRAC x the canvas width) and the clustered row is centred.
MAX_GAP_MULT = 3.0
MAX_GAP_FRAC = 0.08

# Importance tiers (v5): when even the floors do not fit, floors are lowered
# tier by tier from the least important items up (water-filling: everyone
# keeps a minimum, the headline / logo / CTA keep theirs and take the room
# that the search then finds). RESCUE_STEPS is tried in order until the
# floors fit; the primary tier is never lowered (an overflowing plan with a
# readable headline beats one with a 4 px headline).
TIER_PRIMARY, TIER_HERO, TIER_SECONDARY, TIER_MINOR = 0, 1, 2, 3
RESCUE_STEPS = ((TIER_MINOR, 0.55), (TIER_SECONDARY, 0.55), (TIER_MINOR, 0.55), (TIER_HERO, 0.55),
                (TIER_SECONDARY, 0.55))
ABS_MIN_FONT = 4.0          # rescue mode never lowers a text floor below this (a floor that was lower stays)


# Type scale and whitespace (v11, engine/flags TYPESCALE). Secondary copy keeps
# a hierarchy under the headline (never closer than TYPE_RATIO_MIN, or the
# source's own ratio when that was smaller; never further than TYPE_RATIO_MAX
# or the source's ratio when that was larger), body copy wraps at a readable
# measure (MEASURE_MAX_CHARS average characters per line), and the `air`
# variant reserves AIR_SHARE of the slot height as whitespace before the k
# search hands out the room.
STRIP_MAX_H = 120.0
STRIP_MARGIN = 0.035        # v11 strip composer: margin and gap as shares of the short side
STRIP_GAP = 0.03
TRACKING_STEPS = (0.5, 0.25, 0.0)
LEADING_MIN = 0.95          # ... and line height down to this before the font goes under its floor
STRIP_WORD_ROOM = 0.9       # ... measured against this share of the strip's inner width (a column's text width)   # ... letter spacing given up in these steps before a text goes under its floor
SECONDARY_IMAGE_MIN = 0.06  # v11 (UNITS): a picture at least this share of the canvas gets its own slot
TYPE_RATIO_MIN = 1.25
TYPE_RATIO_MAX = 2.6
MEASURE_MAX_CHARS = 75
AIR_SHARE = 0.12


@dataclass(frozen=True)
class PlanOptions:
    """Knobs that turn one floor plan into several candidates (see
    engine/candidates.py). All default to the v1 behaviour.

    hero_side   auto | left | right | stack
                where the hero goes: on the side it had in the source, forced
                to one side (a square canvas then uses the wide plan), or inside
                the column (a wide canvas then uses the column plan). Banners:
                left = right after the logo, right = at the far end of the row.
    spacing     multiplier on the canvas margin and the gap (0.75 tight, 1.3 loose)
    scale_bias  multiplier on the ideal (k = 1) sizes of everything
    chips       row | corner | beside_cta
                badges & stickers: a row in the column (v1), pinned to the
                canvas corners they came from, or in one row with the CTA
    text_flow   stacked | inline | column
                banner text: headline above the rest (v1), beside it, or
                (column) the whole content in one column beside the hero,
                i.e. the wide plan on a banner canvas (v4)
    fill        uniform shrink speeds: every item grows or shrinks at the same
                rate, so the k search becomes a single global-scale search
                (gk7494 / tailsgit style) on `scale_bias` x the proportional
                size; with the default speeds minor items give up room first
    """

    hero_side: str = "auto"
    spacing: float = 1.0
    scale_bias: float = 1.0
    chips: str = "row"
    text_flow: str = "stacked"
    fill: bool = False
    air: bool = False           # v11 (TYPESCALE): AIR_SHARE of the column height is reserved as whitespace
    compose: str = ""           # v11 compositions mined from the designer references (engine/flags COMPOSE):
                                # bookend (banner: logo and CTA at the two ends, the rest centred between),
                                # bleed (the hero flush with its side edge, full height), twotier (banner:
                                # one-line headline above a row of copy / chips / CTA), header (wide: the
                                # headline across the top, hero beside the rest below)
    rescue: bool = False        # v9 (critic recipe): secondary copy yields first; the secondary and minor
                                # floors start lowered by the first two RESCUE_STEPS, so the k search hands
                                # the room to the headline / logo / CTA on a canvas where not everything can
                                # be readable at once (a 160 px strip holding five paragraphs)

    @property
    def name(self) -> str:
        """Only the knobs that differ from the defaults."""
        parts = []
        for f in ("hero_side", "chips", "text_flow", "spacing", "scale_bias"):
            v, d = getattr(self, f), getattr(DEFAULT_OPTIONS, f)
            if v != d:
                parts.append(f"{f.replace('_side', '').replace('_flow', '').replace('_bias', '')}={v:g}" if isinstance(v, float) else f"{f.replace('_side', '').replace('_flow', '')}={v}")
        if self.compose:
            parts.append(f"compose={self.compose}")
        if self.air:
            parts.append("air")
        if self.fill:
            parts.append("fill")
        if self.rescue:
            parts.append("rescue")
        return ", ".join(parts) or "default"


DEFAULT_OPTIONS = PlanOptions()


def is_strip(target: Box) -> bool:
    """A tiny format where every pixel of the short side counts (v11): a tall
    strip (w / h <= STRIP_MAX_RATIO) or a banner at most STRIP_MAX_H tall."""
    return target.w / max(target.h, 1.0) <= STRIP_MAX_RATIO or (
        target.w / max(target.h, 1.0) > WIDE_MAX_RATIO and target.h <= STRIP_MAX_H)


def ratio_family(target: Box) -> str:
    r = target.w / target.h
    if r <= TALL_MAX_RATIO:
        return TALL
    if r <= SQUARE_MAX_RATIO:
        return SQUARE
    if r <= WIDE_MAX_RATIO:
        return WIDE
    return BANNER


# ---------------------------------------------------------------------------
# canvas metrics


@dataclass
class Metrics:
    target: Box
    margin: float
    gap: float
    cap_scale: float      # font multiplier at k = 1 (relative to the source font_size)
    short: bool           # short canvas: minimum font sizes scale with the height
    prop: float = 1.0     # the source-proportional scale, sqrt(target area / source area)
    bias: float = 1.0     # PlanOptions.scale_bias: ambition on top of the proportional size

    @classmethod
    def for_canvas(cls, src: Box, target: Box, family: str, spacing: float = 1.0,
                   scale_bias: float = 1.0) -> "Metrics":
        mn = min(target.w, target.h)
        margin = clamp(0.05 * mn * spacing, 4.0, 60.0)
        gap = clamp(0.035 * mn * spacing, 4.0, 48.0)
        if target.w / max(target.h, 1.0) <= STRIP_MAX_RATIO:
            from engine import flags
            if flags.enabled("strip"):
                # v11 strip composer: on a 160 px skyscraper the margins are what the
                # headline needs to stay readable (on a 90 px leaderboard the tighter
                # margins pushed one layout into a collision and fixed nothing)
                margin = clamp(STRIP_MARGIN * mn * spacing, 3.0, 60.0)
                gap = clamp(STRIP_GAP * mn * spacing, 3.0, 48.0)
        prop = clamp(math.sqrt(target.area / src.area), 0.2, 2.5)
        cap = prop
        if family in (WIDE, BANNER):
            cap *= 1.15   # horizontal layouts give each line more room
        cap *= scale_bias
        return cls(target=target, margin=margin, gap=gap, cap_scale=cap, short=target.h < SHORT_CANVAS,
                   prop=prop, bias=scale_bias)

    def min_font(self, role: str) -> float:
        """Smallest font size we are willing to emit for a role: an absolute
        floor for normal canvases (scaled with the height below SHORT_CANVAS)
        and, on big canvases, a floor relative to the long side so that a
        12 px line on a 1920 px story does not count as readable."""
        h = self.target.h
        if role == HEADLINE:
            base = 12.0 if not self.short else _round(clamp(0.11 * h, 8.0, 12.0))
        elif role == CTA:
            base = 12.0 if not self.short else _round(clamp(0.09 * h, 7.0, 12.0))
        elif role == FINE_PRINT:
            base = 9.0 if not self.short else _round(clamp(0.065 * h, 5.0, 9.0))
        elif role == BADGE:
            # On a 90px banner a badge is a coloured pill; its label is a hint.
            base = 8.0 if not self.short else _round(clamp(0.035 * h, 3.0, 8.0))
        else:
            base = 10.0 if not self.short else _round(clamp(0.08 * h, 6.0, 10.0))
        rel = max(5.0, FONT_FLOOR_LONG * max(self.target.w, self.target.h)) * FONT_FLOOR_MULT.get(role, 1.0)
        if self.target.w / max(self.target.h, 1.0) <= STRIP_MAX_RATIO:
            rel = max(rel, STRIP_FONT_FRAC.get(role, STRIP_FONT_FRAC_DEFAULT) * self.target.w)
        return max(base, _round(rel))

    @property
    def short_side(self) -> float:
        return min(self.target.w, self.target.h)


def _round(v: float) -> float:
    return float(int(v + 0.5))


# ---------------------------------------------------------------------------
# layout items: each knows its size at a shrink factor k and how to place itself


class _Item:
    prio: float = PRIO_PRIMARY
    tier: int = TIER_PRIMARY   # importance tier for rescue mode (independent of the shrink speed)
    w_frac: float = 1.0        # share of the available width this item may use
    applies_w_frac: bool = False   # True: size() applies w_frac itself; else the caller does
    w: float = 0.0
    h: float = 0.0

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        raise NotImplementedError

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        raise NotImplementedError

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        """Rescue mode: lower the floors of this item by `factor` when its
        importance tier is `tier` or lower (bigger number). Primary items
        (headline, logo, CTA) are never asked."""
        return None


def tier_of(prio: float) -> int:
    """Importance tier from the nominal shrink speed (before `fill` makes
    every speed uniform)."""
    if prio >= PRIO_MINOR:
        return TIER_MINOR
    if prio >= PRIO_SECONDARY:
        return TIER_SECONDARY
    if prio >= PRIO_HERO:
        return TIER_HERO
    return TIER_PRIMARY


BUTTON_LABEL_INSIDE = 0.8   # a label is "in" its pill when this much of its box is (analyze.INSIDE_FRAC)
MIN_TEXT_BOX_H = 1.8        # a text box never drops below this height (a 1 px font needs ~1.4 px)
RESCUE_FACTOR = 0.55        # kept for reference: v3's single rescue step (v5 uses RESCUE_STEPS, tier by tier)


class _Scaled(_Item):
    """An image, a group, or any unit that moves as one piece: uniform scale.

    A button-like group (shapes + one label, no images) whose uniform scale
    would push the label below `label_floor` is *squeezed* instead: the
    label is set at the readable size and the pill is re-shaped around it,
    insets scaled with the font and trimmed to the slot. That is what a
    designer does with a wide button on a narrow canvas."""

    applies_w_frac = True

    def __init__(self, unit: Unit, elements: dict[str, dict[str, Any]], ideal_s: float,
                 floor_s: float, prio: float = PRIO_PRIMARY, w_frac: float = 1.0,
                 m: TextMeasurer | None = None, label_floor: float = 0.0, tier: int | None = None):
        self.unit = unit
        self.elements = elements
        self.tier = tier_of(prio) if tier is None else tier
        self.cap_s = float("inf")      # ceiling on the scale (set by the builder; `grow` respects it)
        self.ext_w, self.ext_h = _extents(unit, elements)
        # every member box stays >= 1 px (the official renderer rejects a 0 px
        # text box) and every text member tall enough for a 1 px font to fit
        thin = min((min(b.w, b.h) for b in unit.members.values() if min(b.w, b.h) > 0), default=1.0)
        text_h = min((unit.members[e].h for e in unit.ids if elements[e]["type"] == "text" and unit.members[e].h > 0),
                     default=0.0)
        self.min_s = max(1e-3, 1.0 / thin, (MIN_TEXT_BOX_H / text_h) if text_h else 0.0)
        self.ideal_s = max(ideal_s, floor_s)
        self.floor_s = floor_s
        self.prio = prio
        self.w_frac = w_frac
        self.s = 1.0
        self.m = m
        self.anchor_unit: Unit | None = None    # row parts: the row decides the corner
        self.label_floor = label_floor
        self.label = _button_label(unit, elements) if (m is not None and label_floor > 0) else None
        self.squeezed = False
        self.sq: tuple = ()
        from engine import flags
        self.strip = flags.enabled("strip") and m is not None   # v11: tracking may give way in a squeeze
        self.tracking = 1.0

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        s = max(self.floor_s, self.ideal_s * k ** self.prio)
        s = min(s, aw * self.w_frac / self.ext_w, ah / self.ext_h)   # never larger than the slot
        self.s = max(s, self.min_s)
        self.squeezed = False
        self.tracking = 1.0
        if self.label is not None:
            src_font = float(self.elements[self.label]["font_size"])
            if src_font * self.s < self.label_floor - 1e-6 and self._squeeze(k, aw, ah):
                return self.w, self.h
        self.w, self.h = self.ext_w * self.s, self.ext_h * self.s
        return self.w, self.h

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        if self.tier >= tier:
            self.floor_s *= factor
            self.label_floor = max(min(ABS_MIN_FONT, self.label_floor), self.label_floor * factor)

    def grow(self, max_h: float, max_w: float) -> None:
        """Take leftover room (v5): scale up to `max_h` x `max_w`, never past
        the builder's ceiling (`cap_s`); a squeezed button stays as it is."""
        if self.squeezed:
            return
        s = min(max_h / self.ext_h, max_w / self.ext_w, self.cap_s)
        if s > self.s + 1e-6:
            self.s = s
            self.w, self.h = self.ext_w * s, self.ext_h * s

    def _squeeze(self, k: float, slot_w: float, slot_h: float) -> bool:
        el = self.elements[self.label]
        src_font = float(el["font_size"])
        ub, lb = self.unit.box, self.unit.members[self.label]
        ink_src = self.m.layout(el, src_font, max(lb.w, 1.0)).max_width
        align = str(el.get("text_align", "left") or "left")
        if align == "center":
            ink_x = lb.x + (lb.w - ink_src) / 2.0
        elif align == "right":
            ink_x = lb.right - ink_src
        else:
            ink_x = lb.x
        pad_l, pad_r = max(0.0, ink_x - ub.x), max(0.0, ub.right - (ink_x + ink_src))
        pad_t, pad_b = max(0.0, lb.y - ub.y), max(0.0, ub.bottom - lb.bottom)
        wanted = max(self.label_floor, src_font * max(self.floor_s, self.ideal_s * k ** self.prio))
        font = max(1.0, float(int(wanted)))

        def attempt(font: float, max_lines: int, strict: bool) -> bool:
            s = font / src_font
            min_pad = 0.5 * font
            avail = slot_w - 2.0 * min_pad
            lay = self.m.layout(el, font, max(1.0, avail))
            if lay.max_width > avail + 0.5 or lay.n_lines > max_lines:
                return False
            ink_w = lay.max_width
            room = max(0.0, (slot_w - ink_w) / 2.0)
            pl = max(min_pad, min(pad_l * s, room))
            pr = max(min_pad, min(pad_r * s, room))
            text_h = math.ceil(lay.height) + 1.0
            w, h = ink_w + pl + pr, text_h + (pad_t + pad_b) * s
            if strict and (w > slot_w + 0.5 or h > slot_h + 0.5):
                return False
            self.squeezed = True
            self.sq = (font, pl, pr, pad_t * s, ink_w, text_h)
            self.w, self.h = w, h
            return True

        while True:
            if attempt(font, 2, True):
                return True
            if font <= self.label_floor:
                break
            font -= 1.0
        # At the floor: a long label may take three lines; as a last resort
        # the pill is reshaped at the floor font whatever its height (the
        # plan's shrink search and rescue mode make room, a 2 px label never reads).
        if attempt(self.label_floor, 3, True) or attempt(self.label_floor, 4, False):
            return True
        # v11 strip composer: a tracked label gives up its letter spacing
        # before the button gives up its readable size.
        ls = float(el.get("letter_spacing", 0.0) or 0.0)
        if self.strip and ls > 0.0:
            for f in TRACKING_STEPS:
                el = dict(self.elements[self.label], letter_spacing=ls * f)
                if attempt(self.label_floor, 3, True) or attempt(self.label_floor, 4, False):
                    self.tracking = f
                    return True
        return False

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        box = self.unit.box
        if self.squeezed:
            font, pl, pr, pt, ink_w, text_h = self.sq
            pill = Box(x + (w - self.w) / 2.0, y + (h - self.h) / 2.0, self.w, self.h)
            sx, sy = pill.w / max(box.w, 1e-6), pill.h / max(box.h, 1e-6)
            src_font = float(self.elements[self.label]["font_size"])
            for eid, mb in self.unit.members.items():
                if eid == self.label:
                    out.boxes[eid] = Box(pill.x + pl - 0.5, pill.y + pt, ink_w + 1.0, text_h)
                    out.font_size[eid] = font
                    out.font_scale[eid] = font / src_font
                    if self.tracking < 1.0:
                        out.tracking[eid] = self.tracking
                else:
                    out.boxes[eid] = Box(pill.x + (mb.x - box.x) * sx, pill.y + (mb.y - box.y) * sy,
                                         mb.w * sx, mb.h * sy)
                    out.font_scale[eid] = sy
            return
        bw, bh = box.w * self.s, box.h * self.s
        new = Box(x + (w - bw) / 2.0, y + (h - bh) / 2.0, bw, bh)
        _place_members(self.unit, new, self.s, out)
        for eid in self.unit.ids:
            out.font_scale[eid] = self.s
            el = self.elements[eid]
            if el["type"] == "text":
                out.font_size[eid] = max(1.0, float(el["font_size"]) * self.s)


def _button_label(unit: Unit, elements: dict[str, dict[str, Any]]) -> str | None:
    """The label id of a button-like group: shapes plus exactly one text that
    sits inside them, and no images. Anything else is not squeezable."""
    if not unit.is_group:
        return None
    texts = [e for e in unit.ids if elements[e]["type"] == "text"]
    shapes = [e for e in unit.ids if elements[e]["type"] == "shape"]
    if len(texts) != 1 or not shapes or len(texts) + len(shapes) != len(unit.ids):
        return None
    lb = unit.members[texts[0]]
    hull = Box.union(unit.members[s] for s in shapes)
    if hull.contains_fraction(lb) < BUTTON_LABEL_INSIDE:
        return None
    return texts[0]


class _TextItem(_Item):
    """A single text element whose font size is fitted to its slot.

    The footprint the plan reserves (`w` / `h`) is the *painted* extent of
    the text (its glyphs, not its box: the box starts `ascent - cap height`
    above the first caps and may be wider than the longest line), together
    with the shapes attached to the text (an underline bar, a divider): each
    keeps its position relative to the painted extent in units of the font
    size, so a bar under a heading stays under it at every size and line
    count. `place` puts the text box where that footprint lands.

    A text rotated to (near) vertical is wrapped and fitted in its *own*
    frame: the slot's height is its line width budget and the slot's width
    its height budget; its footprint is the rotated extent of its box and
    attachments scale with it uniformly."""

    def __init__(self, unit: Unit, el: dict[str, Any], m: TextMeasurer, cap: float,
                 floor: float, prio: float, natural_width: bool = False, hard_w: float = 0.0,
                 tier: int | None = None):
        self.unit = unit
        self.el = el
        self.m = m
        self.floor = float(floor)
        self.cap = max(float(int(cap)), self.floor)
        self.prio = prio
        self.tier = tier_of(prio) if tier is None else tier
        self.len_share = 1.0        # vertical text: share of the slot height its lines may run along
        self.natural_width = natural_width
        self.hard_w = float(hard_w)     # a line never wider than this (the canvas), floor or not
        self.font = self.cap
        self.ink_w = 0.0            # widest wrapped line at the last size() (footprint width)
        self.rotation = float(el.get("rotation", 0.0) or 0.0)
        self.vertical = is_vertical(self.rotation)
        self.own_w = self.own_h = 0.0   # the box in the element's own frame
        self.box_off = (0.0, 0.0)       # text box top-left relative to the footprint top-left
        self.att: list[tuple[str, float, float, float, float, str, str]] = []   # attachments (see _attach)
        self.att_boxes: dict[str, Box] = {}   # attachment boxes relative to the footprint at the last size()
        from engine import flags
        self.clip_att = flags.enabled("strip")   # v11: attachments clipped to the slot's width
        self.tracking = 1.0                   # v11 strip composer: letter-spacing multiplier (self.el has it applied)
        self.leading: float | None = None     # ... and a tighter line_height (self.el has it applied)
        self.lead: _TextItem | None = None    # v11 type scale: the headline this text stays under
        self.ratio_lo = self.ratio_hi = 0.0   # ... font within [lead / ratio_hi, lead / ratio_lo]
        self.measure_w = 0.0                  # v11: widest line a body text may set (0 = none)
        self._attach()

    def _attach(self) -> None:
        """Record each attached shape relative to the source text's painted
        extent: x from the aligned edge, y from the painted top or bottom
        (whichever it is closer to), size, all in font-size units."""
        src_font = max(1.0, float(self.el["font_size"]))
        sb = self.unit.members[self.unit.primary or self.unit.ids[0]]
        painted = self.m.painted(self.el, src_font, max(sb.w, 1.0)).box(sb)
        align = str(self.el.get("text_align", "left") or "left")
        for eid in self.unit.attachments:
            ab = self.unit.members[eid]
            if align == "center":
                dx = ab.cx - painted.cx
            elif align == "right":
                dx = ab.right - painted.right
            else:
                dx = ab.x - painted.x
            if ab.cy <= painted.cy:
                dy, vref = ab.y - painted.y, "top"
            else:
                dy, vref = ab.y - painted.bottom, "bottom"
            self.att.append((eid, dx / src_font, dy / src_font, ab.w / src_font, ab.h / src_font, align, vref))

    def font_at(self, k: float) -> float:
        f = max(self.floor, float(int(self.cap * k ** self.prio)))
        if self.lead is not None:
            lead = self.lead.font_at(k)
            f = min(f, float(int(lead / self.ratio_lo)))
            f = max(f, float(int(lead / self.ratio_hi)))
            f = max(self.floor, f)
        return f

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        if self.tier >= tier:
            self.floor = max(min(ABS_MIN_FONT, self.floor), float(int(self.floor * factor)))

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        # Budgets in the element's own frame. A vertical text runs along the
        # slot's height; in a column it may take only `len_share` of it and
        # wraps onto a second (third) line beyond that, like a spine label.
        bw, bh = (ah * self.len_share, aw) if self.vertical else (aw, ah)
        font = self.font_at(k)
        if self.measure_w > 0.0 and not self.vertical:
            bw = min(bw, max(self.measure_w * font, 1.0))
        return self._size_at(font, bw, bh)

    def rail_size(self, k: float, thick_budget: float, len_budget: float) -> tuple[float, float]:
        """A vertical text beside the hero (v5): one line along `len_budget`
        at the largest font from font_at(k) down to the floor; it wraps only
        when the single line is still longer at the floor."""
        font = self.font_at(k)
        while font > self.floor and _single_line_width(self.m, self.el, font) + 2.0 > len_budget:
            font = max(self.floor, font - 1.0)
        return self._size_at(font, len_budget, thick_budget)

    def _size_at(self, font: float, bw: float, bh: float) -> tuple[float, float]:
        # An over-wide single word overflows the box in the renderer: shrink first.
        while font > self.floor and self.m.layout(self.el, font, bw).max_width > bw + 0.5:
            font = max(self.floor, font - 1.0)
        # Below the floor only when the line would not even fit the canvas
        # (a letter-spaced heading on a 160 px skyscraper): the fitter would
        # shrink it anyway after the box is clamped, but then its attached
        # bars would keep the offsets planned for the bigger font.
        # Measured at the slot's wrap width: a letter-spaced line spills less
        # when it breaks earlier, so the canvas is the limit, not the budget.
        if self.hard_w > 0 and not self.vertical:
            while font > 1.0 and self.m.layout(self.el, font, bw).max_width > self.hard_w + 0.5:
                font -= 1.0
        natural = self.natural_width or self.vertical
        while True:
            w = min(bw, _single_line_width(self.m, self.el, font) + 2.0) if natural else bw
            lay = self.m.layout(self.el, font, w)
            h = math.ceil(lay.height) + 1.0
            if not self.vertical or h <= bh + 0.5 or font <= self.floor:
                break
            font = max(self.floor, font - 1.0)    # a vertical label too thick for its slot
        if lay.max_width > w + 0.5:
            # Still wider than the budget at the floor (letter-spaced line: the
            # renderer wraps on glyph widths and the spacing spills over): the
            # box takes the real width so the font the plan chose really fits
            # and the plan sees the true footprint.
            w = lay.max_width + 1.0
        elif not self.rotation:
            # The box hugs the wrapped lines (v5): greedy wrapping at the widest
            # line's width reproduces the same lines, and a box no wider than
            # its ink cannot start off canvas when the ink is placed at the
            # slot's edge (the rectifier then dragged box and ink into the
            # neighbour). The renderer wraps on glyph widths without letter
            # spacing (a negatively spaced line needs its glyph width) and
            # counts the space after a line's last word before it breaks, so
            # the box keeps room for one space.
            hug = lay.max_width
            if scaled_letter_spacing(self.el, font) < 0.0:
                hug = self.m.layout(dict(self.el, letter_spacing=0.0), font, w).max_width
            space = self.m.layout(dict(self.el, content=" ", letter_spacing=0.0), font, 1e7).max_width
            w = max(1.0, min(w, hug + space + 1.0))
        self.font = font
        self.own_w, self.own_h = w, h
        self.att_boxes = {}
        if self.rotation:
            ext = _rotated_extent(w, h, self.rotation)
            self.w, self.h = ext
            self.ink_w = self.w if self.vertical else min(self.w, lay.max_width)
            self.box_off = ((self.w - w) / 2.0, (self.h - h) / 2.0)
            return self.w, self.h
        painted = self.m.painted(self.el, font, w)
        pb = Box(painted.x, painted.top, painted.w, painted.h)      # relative to the box
        boxes = [pb]
        rel: dict[str, Box] = {}
        room = max(1.0, bh - pb.h)      # height the slot leaves for attachments above / below the text (v5)
        for eid, dx, dy, aw_f, ah_f, align, vref in self.att:
            a_w, a_h = min(aw_f * font, max(w, 1.0)), min(ah_f * font, room)
            if align == "center":
                ax = pb.cx + dx * font - a_w / 2.0
            elif align == "right":
                ax = pb.right + dx * font - a_w
            else:
                ax = pb.x + dx * font
            ay = (pb.y if vref == "top" else pb.bottom) + dy * font
            # An attachment much taller than its text (a bar that ran along
            # a whole list) never makes the footprint taller than the slot.
            if vref == "bottom":
                ay = min(ay, pb.bottom + room - a_h)
            else:
                ay = max(ay, pb.bottom - bh)
            if self.clip_att:
                # v11 strip composer: a bar never makes the footprint wider than
                # the slot (an 84 px rule beside a 145 px headline on a 160 px
                # skyscraper pushed the pair off canvas and the rectifier halved
                # the headline); it is clipped to the slot centred on the text
                span = max(bw, pb.w)
                lo = pb.cx - span / 2.0
                x0, x1 = max(ax, lo), min(ax + a_w, lo + span)
                if x1 - x0 < 1.0:
                    x0 = clamp(ax, lo, lo + span - 1.0)
                    x1 = x0 + max(1.0, min(a_w, span))
                ax, a_w = x0, x1 - x0
            rel[eid] = Box(ax, ay, a_w, a_h)
            boxes.append(rel[eid])
        foot = Box.union(boxes)
        self.w, self.h = foot.w, foot.h
        self.ink_w = painted.w
        self.box_off = (-foot.x, -foot.y)
        self.att_boxes = {eid: Box(b.x - foot.x, b.y - foot.y, b.w, b.h) for eid, b in rel.items()}
        return self.w, self.h

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        eid = self.unit.primary or self.unit.ids[0]
        src_font = max(1.0, float(self.el["font_size"]))
        align = str(self.el.get("text_align", "left") or "left")
        if align == "right":
            fx = x + w - self.w
        elif align == "center" or self.rotation:
            fx = x + (w - self.w) / 2.0
        else:
            fx = x
        fy = y + (h - self.h) / 2.0     # footprint top-left
        if self.rotation:
            cx, cy = x + w / 2.0, y + h / 2.0
            box = Box(cx - self.own_w / 2.0, cy - self.own_h / 2.0, self.own_w, self.own_h)
            out.boxes[eid] = box
            s = self.font / src_font
            sb = self.unit.members[eid]
            for aid in self.unit.attachments:
                ab = self.unit.members[aid]
                out.boxes[aid] = Box(box.x + (ab.x - sb.x) * s, box.y + (ab.y - sb.y) * s, ab.w * s, ab.h * s)
                out.font_scale[aid] = s
        else:
            out.boxes[eid] = Box(fx + self.box_off[0], fy + self.box_off[1], self.own_w, self.own_h)
            for aid, b in self.att_boxes.items():
                out.boxes[aid] = Box(fx + b.x, fy + b.y, b.w, b.h)
                out.font_scale[aid] = self.font / src_font
        out.font_size[eid] = self.font
        out.font_scale[eid] = self.font / src_font
        if self.tracking < 1.0:
            out.tracking[eid] = self.tracking
        if self.leading is not None:
            out.leading[eid] = self.leading


def is_vertical(rotation: float) -> bool:
    """Rotated closer to vertical than to horizontal (|rotation| near 90)."""
    r = abs(float(rotation or 0.0)) % 180.0
    return VERTICAL_MIN_DEG <= r <= 180.0 - VERTICAL_MIN_DEG


def _rotated_extent(w: float, h: float, rotation: float) -> tuple[float, float]:
    a = math.radians(rotation)
    return (w * abs(math.cos(a)) + h * abs(math.sin(a)), w * abs(math.sin(a)) + h * abs(math.cos(a)))


class _TextColumn(_Item):
    """Headline + subhead (+ body, fine print) stacked in a column whose width
    is chosen so the headline wraps on at most `max_lines` lines and the
    whole stack fits the available height. Used by the banner plan.

    v5: when even the widest single column is taller than the strip, the
    texts flow into several columns side by side (reading order kept, each
    column as tall as the strip allows) instead of shrinking to nothing: a
    90 px banner holds the headline in one column and the rest beside it."""

    applies_w_frac = True

    def __init__(self, texts: list[_TextItem], head: _TextItem, gap: Callable[[float], float],
                 w_frac: float, max_lines: int = 2, flow: bool = True):
        self.texts = texts
        self.head = head
        self.gap = gap
        self.w_frac = w_frac
        self.max_lines = max_lines
        self.over = False
        self.flow = flow
        self.columns: list[list[_TextItem]] = [list(texts)]
        self.col_w: list[float] = []

    def _height(self, k: float, w: float, ah: float) -> float:
        total = 0.0
        for t in self.texts:
            total += t.size(k, w, ah)[1]
        return total + self.gap(self.head.font) * (len(self.texts) - 1)

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        max_w = max(1.0, aw * self.w_frac)
        head_font = self.head.font_at(k)
        lo = _min_width_for_lines(self.head.m, self.head.el, head_font, self.max_lines, max_w)
        for t in self.texts:
            if t.vertical:
                continue   # its footprint width is its line height; size() handles it
            lo = max(lo, min(max_w, _longest_word_width(t.m, t.el, t.font_at(k)) + 1.0))
        lo = min(lo, max_w)
        self.columns = [list(self.texts)]
        self.col_w = []
        if self._height(k, lo, ah) <= ah + 0.5:
            w = lo
            self.over = False
        else:
            a, b = lo, max_w
            if self._height(k, b, ah) > ah + 0.5:
                w = max_w
                self.over = True
                if self.flow and len(self.texts) > 1:
                    return self._flow(k, lo, max_w, ah)
            else:
                for _ in range(12):
                    mid = (a + b) / 2.0
                    if self._height(k, mid, ah) <= ah + 0.5:
                        b = mid
                    else:
                        a = mid
                w = b
                self.over = False
        self.h = self._height(k, w, ah)
        self.w = self._ink(w)
        return self.w, self.h

    def _flow(self, k: float, lo: float, max_w: float, ah: float) -> tuple[float, float]:
        """Pack the texts into the fewest columns whose stacks fit `ah`, each
        text at the column width `lo` (the narrowest width the headline's
        line policy allows); a text taller than the strip on its own gets a
        column of its own and overflows it (the plan's k search shrinks it)."""
        w = max(1.0, lo)
        g = self.gap(self.head.font)
        cols: list[list[_TextItem]] = [[]]
        run = 0.0
        for t in self.texts:
            h = t.size(k, w, ah)[1]
            add = h + (g if cols[-1] else 0.0)
            if cols[-1] and run + add > ah + 0.5:
                cols.append([t])
                run = h
            else:
                cols[-1].append(t)
                run += add
        self.columns = cols
        self.col_w = [max(min(w, max((t.ink_w for t in c), default=w) + 1.0), max(t.w for t in c)) for c in cols]
        heights = [sum(t.h for t in c) + g * (len(c) - 1) for c in cols]
        self.over = max(heights) > ah + 0.5
        self.w = sum(self.col_w) + 2.0 * g * (len(cols) - 1)
        self.h = max(heights)
        return self.w, self.h

    def _ink(self, wrap_w: float) -> float:
        """Box width that keeps the wrap found at `wrap_w` but hugs the glyphs,
        so the gaps a viewer sees are the gaps the plan intended."""
        ink = max((t.ink_w for t in self.texts), default=wrap_w)
        return max(min(wrap_w, ink + 1.0), max((t.w for t in self.texts), default=0.0))

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        for t in self.texts:
            t.relax(factor, tier)

    def comfortable_width(self, k: float, max_w: float) -> float:
        """Width at which the headline sits on one line and the rest on two."""
        best = 0.0
        for t in self.texts:
            lines = 1 if t is self.head else 2
            best = max(best, _min_width_for_lines(t.m, t.el, t.font_at(k), lines, max_w))
        return min(best, max_w)

    def widen(self, k: float, w: float, ah: float) -> None:
        if w > self.w and len(self.columns) == 1:
            self.h = self._height(k, w, ah)
            self.w = self._ink(w)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        g = self.gap(self.head.font)
        if len(self.columns) == 1:
            cy = y + (h - self.h) / 2.0
            for t in self.texts:
                t.place(x + (w - self.w) / 2.0, cy, self.w, t.h, out)
                cy += t.h + g
            return
        cx = x + (w - self.w) / 2.0
        for col, cw in zip(self.columns, self.col_w):
            ch = sum(t.h for t in col) + g * (len(col) - 1)
            cy = y + (h - ch) / 2.0
            for t in col:
                t.place(cx, cy, cw, t.h, out)
                cy += t.h + g
            cx += cw + 2.0 * g


class _Chips(_Item):
    """Badges, secondary images and other small stuff packed into a cluster:
    rows that wrap on the available width, columns that wrap on the
    available height, or "auto" (v4): rows first, and columns only when the
    wrapped rows would not fit the available height (a banner's chips sit
    in a row like the designer's, and stack only when the strip is too
    short for that)."""

    applies_w_frac = True

    def __init__(self, chips: list[_Scaled], direction: str, gap: float, w_frac: float = 1.0):
        self.chips = chips
        self.direction = direction
        self.gap = gap
        self.w_frac = w_frac
        self.prio = PRIO_MINOR
        self.offsets: list[tuple[float, float]] = []

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        aw = aw * self.w_frac
        for c in self.chips:
            c.size(k, aw, ah)
        if self.direction == "auto":
            self._pack("row", aw, ah)
            if self.h > ah + 0.5:
                self._pack("col", aw, ah)
            return self.w, self.h
        self._pack(self.direction, aw, ah)
        return self.w, self.h

    def _pack(self, direction: str, aw: float, ah: float) -> None:
        g = self.gap
        lines: list[list[_Scaled]] = [[]]
        run = 0.0
        for c in self.chips:
            extent = c.w if direction == "row" else c.h
            limit = aw if direction == "row" else ah
            add = extent + (g if lines[-1] else 0.0)
            if lines[-1] and run + add > limit + 0.5:
                lines.append([c])
                run = extent
            else:
                lines[-1].append(c)
                run += add
        self.offsets = []
        if direction == "row":
            widths = [sum(c.w for c in ln) + g * (len(ln) - 1) for ln in lines]
            heights = [max(c.h for c in ln) for ln in lines]
            self.w = max(widths)
            self.h = sum(heights) + g * (len(lines) - 1)
            y = 0.0
            for ln, lw, lh in zip(lines, widths, heights):
                x = (self.w - lw) / 2.0
                for c in ln:
                    self.offsets.append((x, y + (lh - c.h) / 2.0))
                    x += c.w + g
                y += lh + g
        else:
            heights = [sum(c.h for c in ln) + g * (len(ln) - 1) for ln in lines]
            widths = [max(c.w for c in ln) for ln in lines]
            self.h = max(heights)
            self.w = sum(widths) + g * (len(lines) - 1)
            x = 0.0
            for ln, lw, lh in zip(lines, widths, heights):
                y = (self.h - lh) / 2.0
                for c in ln:
                    self.offsets.append((x + (lw - c.w) / 2.0, y))
                    y += c.h + g
                x += lw + g

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        for c in self.chips:
            c.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        x0 = x + (w - self.w) / 2.0
        y0 = y + (h - self.h) / 2.0
        for c, (dx, dy) in zip(self.chips, self.offsets):
            c.place(x0 + dx, y0 + dy, c.w, c.h, out)


class _HeroRails(_Item):
    """The hero with vertical texts running along its sides (v5): the rails
    a poster designer sets beside a photo stay beside it instead of being
    stacked into the column, where their length ate the column's height.
    The rails take their line budget from the hero's height and wrap onto
    a second line when they are longer; the hero gets the width they leave."""

    applies_w_frac = True

    def __init__(self, hero: _Scaled, rails: list[_TextItem], gap: float, hero_box: Box):
        self.hero = hero
        self.rails = rails
        self.gap = gap
        self.prio = hero.prio
        self.tier = hero.tier
        self.unit = hero.unit
        self.sides = {id(r): ("left" if _drawn_extent(r.unit, r.el).cx < hero_box.cx else "right") for r in rails}
        self.k = 1.0

    def _rails_w(self) -> float:
        return sum(r.w + self.gap for r in self.rails)

    def _fit_rails(self, k: float, aw: float, ah: float) -> None:
        # A rail may run a little past the hero (up to RAIL_LEN_SHARE of the
        # column) so that it stays on one line at a readable size.
        budget = max(self.hero.h, RAIL_LEN_SHARE * ah)
        for r in self.rails:
            r.rail_size(k, aw * RAIL_W_FRAC, budget)

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        self.k = k
        self.ah = ah
        rails_w = getattr(self, "_last_rails_w", 0.0)
        for _ in range(2):
            self.hero.size(k, max(1.0, aw - rails_w), ah)
            self._fit_rails(k, aw, ah)
            rails_w = self._rails_w()
        self.hero.size(k, max(1.0, aw - rails_w), ah)
        self._last_rails_w = rails_w
        self.w = self.hero.w + rails_w
        self.h = max([self.hero.h] + [r.h for r in self.rails])
        return self.w, self.h

    def grow(self, max_h: float, max_w: float) -> None:
        self.hero.grow(max_h, max(1.0, max_w - self._rails_w()))
        self._fit_rails(self.k, max_w, getattr(self, "ah", max_h))
        self.w = self.hero.w + self._rails_w()
        self.h = max([self.hero.h] + [r.h for r in self.rails])

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        self.hero.relax(factor, tier)
        for r in self.rails:
            r.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        cx = x + (w - self.w) / 2.0
        left = [r for r in self.rails if self.sides[id(r)] == "left"]
        right = [r for r in self.rails if self.sides[id(r)] != "left"]
        for r in left:
            r.place(cx, y + (h - r.h) / 2.0, r.w, r.h, out)
            cx += r.w + self.gap
        self.hero.place(cx, y + (h - self.hero.h) / 2.0, self.hero.w, self.hero.h, out)
        cx += self.hero.w + self.gap
        for r in right:
            r.place(cx, y + (h - r.h) / 2.0, r.w, r.h, out)
            cx += r.w + self.gap


def _drawn_extent(unit: Unit, el: dict[str, Any]) -> Box:
    """Axis-aligned extent of a unit's primary element as drawn in the source."""
    b = unit.members.get(el["id"], unit.box)
    rot = float(el.get("rotation", 0.0) or 0.0)
    if not rot:
        return b
    w, h = _rotated_extent(b.w, b.h, rot)
    return Box(b.cx - w / 2.0, b.cy - h / 2.0, w, h)


def _beside(text: _TextItem, hero: Unit) -> bool:
    """Did this vertical text run along the hero's side in the source?"""
    ext = _drawn_extent(text.unit, text.el)
    yo = min(ext.bottom, hero.box.bottom) - max(ext.y, hero.box.y)
    if yo < 0.5 * max(ext.h, 1.0):
        return False
    return ext.cx < hero.box.x or ext.cx > hero.box.right


class _Row(_Item):
    """Items side by side (the CTA with the badges, a headline beside its
    subhead), vertically centred. Each child sizes itself on its own share
    of the row's width, so the row never grows wider than its slot."""

    def __init__(self, items: list[_Item], gap: float, shares: list[float] | None = None):
        self.items = items
        self.gap = gap
        self.shares = shares or [1.0 / len(items)] * len(items)
        self.prio = min(it.prio for it in items)

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        inner = max(1.0, aw - self.gap * (len(self.items) - 1))
        self.w = self.gap * (len(self.items) - 1)
        self.h = 0.0
        for it, share in zip(self.items, self.shares):
            w, h = it.size(k, inner * share, ah)
            self.w += w
            self.h = max(self.h, h)
        return self.w, self.h

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        for it in self.items:
            it.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        cx = x + (w - self.w) / 2.0
        for it in self.items:
            it.place(cx, y + (h - it.h) / 2.0, it.w, it.h, out)
            cx += it.w + self.gap


class _Stack(_Item):
    """Items on top of each other (v11 compositions: a one-line headline above
    a row of copy / chips / CTA, a headline across the top of a wide canvas
    above the hero and the rest), aligned left, centred or right."""

    def __init__(self, items: list[_Item], gaps: list[float], align: str = "center", w_frac: float = 1.0):
        self.items = items
        self.gaps = gaps
        self.align = align
        self.w_frac = w_frac
        self.prio = min(it.prio for it in items)

    def size(self, k: float, aw: float, ah: float) -> tuple[float, float]:
        aw = aw * self.w_frac
        self.w, self.h = 0.0, sum(self.gaps)
        for it in self.items:
            w, h = it.size(k, aw if it.applies_w_frac else aw * it.w_frac, ah)
            self.w = max(self.w, w)
            self.h += h
        return self.w, self.h

    def relax(self, factor: float, tier: int = TIER_SECONDARY) -> None:
        for it in self.items:
            it.relax(factor, tier)

    def place(self, x: float, y: float, w: float, h: float, out: Placement) -> None:
        cy = y + (h - self.h) / 2.0
        x0 = x + (w - self.w) / 2.0
        for i, it in enumerate(self.items):
            if self.align == "left":
                ix = x0
            elif self.align == "right":
                ix = x0 + self.w - it.w
            else:
                ix = x0 + (self.w - it.w) / 2.0
            it.place(ix, cy, it.w, it.h, out)
            cy += it.h + (self.gaps[i] if i < len(self.gaps) else 0.0)


# ---------------------------------------------------------------------------
# measuring helpers


def _text_h(m: TextMeasurer, el: dict[str, Any], font: float, width: float) -> float:
    return math.ceil(m.layout(el, font, width).height) + 1.0


def _single_line_width(m: TextMeasurer, el: dict[str, Any], font: float) -> float:
    return m.layout(el, font, 1e7).max_width


def _longest_word_width(m: TextMeasurer, el: dict[str, Any], font: float) -> float:
    lay = m.layout(el, font, 1.0)   # wraps at every space: each line is one word
    return lay.max_width


def _min_width_for_lines(m: TextMeasurer, el: dict[str, Any], font: float,
                         max_lines: int, max_w: float) -> float:
    """Smallest box width at which the text wraps on at most `max_lines`."""
    if el.get("max_lines"):
        max_lines = min(max_lines, int(el["max_lines"]))

    def ok(w: float) -> bool:
        lay = m.layout(el, font, w)
        return lay.n_lines <= max_lines and lay.max_width <= w + 0.5

    if not ok(max_w):
        return max_w
    lo, hi = max(1.0, _longest_word_width(m, el, font)), max_w
    if ok(lo):
        return lo
    for _ in range(12):
        mid = (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _extents(unit: Unit, elements: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """Axis-aligned extents of the unit, allowing for rotated members."""
    rot = 0.0
    for eid in unit.ids:
        rot = max(rot, abs(float(elements[eid].get("rotation", 0.0) or 0.0)))
    w, h = max(unit.box.w, 1.0), max(unit.box.h, 1.0)
    if rot == 0.0:
        return w, h
    a = math.radians(rot)
    return w * abs(math.cos(a)) + h * abs(math.sin(a)), w * abs(math.sin(a)) + h * abs(math.cos(a))


def _largest_k(fits: Callable[[float], bool], items: list[_Item] | None = None) -> tuple[float, bool]:
    """Largest k in [0, 1] for which fits(k) holds (fits is monotone in k).
    Leaves every item sized at the returned k. The flag is False when even
    the floors (k = 0) do not fit; with `items` given, that case first
    tries rescue mode (floors lowered tier by tier, RESCUE_STEPS):
    a readable headline and CTA with small legal copy beat a layout whose
    pieces run into each other."""
    if fits(1.0):
        return 1.0, True
    if not fits(0.0):
        if not items:
            return 0.0, False
        # Water-filling (v5): the least important floors go first, tier by
        # tier, until the floors fit; the search above the floors then hands
        # the room to the headline / logo / CTA first (their shrink speed).
        for tier, factor in RESCUE_STEPS:
            for it in items:
                it.relax(factor, tier)
            if fits(0.0):
                break
        else:
            return 0.0, False
    lo, hi = 0.0, 1.0
    for _ in range(K_SEARCH_STEPS):
        mid = (lo + hi) / 2.0
        if fits(mid):
            lo = mid
        else:
            hi = mid
    fits(lo)
    return lo, True


# ---------------------------------------------------------------------------
# slots: content units sorted into the roles a floor plan knows about


@dataclass
class _Slots:
    logo: Unit | None
    hero: Unit | None
    cta: Unit | None
    texts: list[Unit]          # headline / subhead / body, reading order
    fine: list[Unit]
    chips: list[Unit]          # badges, extra images, shapes, grouped text blocks
    text_align: str            # headline's alignment drives column alignment
    hero_side: str             # left | right (wide plan)
    images: list[Unit] = field(default_factory=list)   # v11 (UNITS): secondary pictures with their own slot


def _collect(analysis: Analysis, elements: dict[str, dict[str, Any]]) -> _Slots:
    logo = hero = cta = None
    texts: list[Unit] = []
    fine: list[Unit] = []
    chips: list[Unit] = []
    for u in sorted(analysis.content_units(), key=lambda u: u.reading_index):
        text_unit = u.primary_type(elements) == "text"   # a text, possibly with attached bars
        if u.role == LOGO and logo is None:
            logo = u
        elif u.role == HERO and hero is None:
            hero = u
        elif u.role == CTA and cta is None:
            cta = u
        elif u.role in (HEADLINE, SUBHEAD, BODY, TEXT) and text_unit:
            texts.append(u)
        elif u.role == FINE_PRINT and text_unit:
            fine.append(u)
        else:
            chips.append(u)
    images: list[Unit] = []
    from engine import flags
    if flags.enabled("units"):
        # v11: a secondary picture that was a real part of the composition
        # (>= SECONDARY_IMAGE_MIN of the canvas) is not a badge-sized chip
        for u in list(chips):
            if u.role == IMAGE and not u.is_row and u.primary_type(elements) == "image" \
                    and u.box.area >= SECONDARY_IMAGE_MIN * analysis.canvas.area:
                chips.remove(u)
                images.append(u)
    head = next((u for u in texts if u.role == HEADLINE), texts[0] if texts else None)
    align = "center"
    if head is not None:
        align = str(elements[head.primary or head.ids[0]].get("text_align", "left") or "left")
        if align not in ("left", "center", "right"):
            align = "center"
    side = "left"
    if hero is not None and head is not None and hero.box.cx > head.box.cx + 0.1 * analysis.canvas.w:
        side = "right"
    return _Slots(logo, hero, cta, texts, fine, chips, align, side, images)


# ---------------------------------------------------------------------------
# item factories


class _Builder:
    def __init__(self, analysis: Analysis, elements: dict[str, dict[str, Any]],
                 met: Metrics, m: TextMeasurer, family: str, uniform: bool = False, rescue: bool = False):
        self.analysis = analysis
        self.elements = elements
        self.met = met
        self.m = m
        self.family = family
        self.slots = _collect(analysis, elements)
        self.target = met.target
        self.uniform = uniform      # PlanOptions.fill: one shrink speed for everything
        self.rescue = rescue        # PlanOptions.rescue: secondary / minor floors pre-lowered (v9)
        from engine import flags
        self.typescale = flags.enabled("typescale")
        self.strip = flags.enabled("strip") and is_strip(met.target)
        self._lead: _TextItem | None = None

    def prio(self, p: float) -> float:
        return PRIO_PRIMARY if self.uniform else p

    # -- text ------------------------------------------------------------

    def text(self, unit: Unit, prio: float, natural_width: bool = False) -> _TextItem:
        el = self.elements[unit.primary or unit.ids[0]]
        cap = float(el["font_size"]) * self.met.cap_scale
        floor = self.met.min_font(unit.role)
        tracking = self._tracking(el, floor) if self.strip else 1.0
        if tracking < 1.0:
            el = dict(el, letter_spacing=float(el["letter_spacing"]) * tracking)
        # tighter leading only in the triage (rescue) variants: applied to every
        # plan it moved one clean leaderboard layout into a collision
        leading = self._leading(el, floor) if self.strip and self.rescue else None
        if leading is not None:
            el = dict(el, line_height=leading)
        it = _TextItem(unit, el, self.m, cap, floor, self.prio(prio), natural_width,
                       hard_w=self.target.w,   # a line may run into the margins before it goes under the floor
                       tier=tier_of(prio))
        it.tracking = tracking
        it.leading = leading
        if self.rescue:
            _pre_relax(it)
        return it

    def _tracking(self, el: dict[str, Any], floor: float) -> float:
        """v11 strip composer: the letter-spacing multiplier at which the
        text's longest word fits the strip's width at its floor font (1.0 =
        unchanged). Tracking is decoration; size is legibility."""
        ls = float(el.get("letter_spacing", 0.0) or 0.0)
        if ls <= 0.0 or is_vertical(float(el.get("rotation", 0.0) or 0.0)):
            return 1.0
        room = STRIP_WORD_ROOM * (self.target.w - 2.0 * self.met.margin)

        def fits(e: dict[str, Any]) -> bool:
            # the renderer wraps on glyph widths and paints the spacing on top:
            # the painted lines at the wrap width must fit, not only the longest word
            return self.m.layout(e, floor, room).max_width <= room + 0.5

        if fits(el):
            return 1.0
        for f in TRACKING_STEPS:
            if fits(dict(el, letter_spacing=ls * f)):
                return f
        return 0.0

    def _leading(self, el: dict[str, Any], floor: float) -> float | None:
        """v11 strip composer: on a short banner a text whose hard line breaks
        alone make it taller than the strip at its floor font sets its lines
        closer (down to LEADING_MIN) before it goes under the floor."""
        if self.target.h > STRIP_MAX_H or is_vertical(float(el.get("rotation", 0.0) or 0.0)):
            return None
        lay = self.m.layout(el, floor, 1e7)
        if lay.n_lines < 2:
            return None
        room = self.target.h - 2.0 * self.met.margin - 2.0
        if lay.height <= room:
            return None
        lh = float(el.get("line_height", 0.0) or 0.0) or (lay.height / max(1, lay.n_lines) / floor)
        new = max(LEADING_MIN, lh * room / lay.height)
        return round(new, 3) if new < lh - 1e-3 else None

    def text_items(self) -> list[_TextItem]:
        items = []
        for u in self.slots.texts:
            prio = PRIO_PRIMARY if u.role == HEADLINE else PRIO_SECONDARY
            items.append(self.text(u, prio))
        self._lead = None
        self._type_scale(items)
        return items

    def fine_items(self) -> list[_TextItem]:
        items = [self.text(u, PRIO_MINOR) for u in self.slots.fine]
        self._type_scale(items)
        return items

    def _type_scale(self, items: list[_TextItem]) -> None:
        """v11 (TYPESCALE): every horizontal secondary text keeps its distance
        under the headline (see TYPE_RATIO_MIN / MAX) and long copy wraps at
        a readable measure. A no-op with the flag off."""
        if not self.typescale:
            return
        head_unit = next((u for u in self.slots.texts if u.role == HEADLINE), None)
        if head_unit is None:
            return
        if self._lead is None or self._lead.unit is not head_unit:
            self._lead = next((it for it in items if it.unit is head_unit), None) or self.text(head_unit, PRIO_PRIMARY)
        lead = self._lead
        if lead.rotation:
            return
        head_src = float(lead.el["font_size"])
        for it in items:
            if it.unit is head_unit or it.rotation:
                continue
            src = max(1.0, float(it.el["font_size"]))
            ratio = head_src / src
            if ratio < 1.0:
                continue          # a text bigger than the headline in the source: not ours to rank
            it.lead = lead
            it.ratio_lo = min(TYPE_RATIO_MIN, ratio)
            it.ratio_hi = max(TYPE_RATIO_MAX, ratio)
            content = str(it.el.get("content", ""))
            if len(content) > MEASURE_MAX_CHARS and it.unit.role != HEADLINE:
                per_char = _single_line_width(it.m, it.el, 100.0) / 100.0 / max(1, len(content))
                it.measure_w = MEASURE_MAX_CHARS * per_char

    def text_gap(self) -> Callable[[float], float]:
        g = self.met.gap
        return lambda head_font: clamp(0.35 * head_font, 3.0, g)

    # -- scaled pieces -----------------------------------------------------

    def _text_floor(self, unit: Unit, role: str) -> float:
        """Scale below which a group's text members go under the role's minimum."""
        fonts = [float(self.elements[e]["font_size"]) for e in unit.ids if self.elements[e]["type"] == "text"]
        if not fonts:
            return 0.0
        return self.met.min_font(role) / max(fonts)

    def scaled(self, unit: Unit, ideal_h: float, floor_h: float, prio: float, w_frac: float,
               role_for_floor: str | None = None, cap_mult: float = 1.0,
               floor_frac: float = 0.0) -> _Scaled:
        """`floor_frac`: the unit's larger dimension never drops below this
        share of the canvas's short side (capped by the slot in size()).
        The ideal is the slot's share of the canvas or, when that is less
        (a 6 % CTA on a 396 px banner), the source-proportional scale times
        the plan's ambition (v4); the cap keeps both under the font cap."""
        ext_w, ext_h = _extents(unit, self.elements)
        wish = max(ideal_h / ext_h, self.met.prop * self.met.bias)
        ideal_s = min(wish, self.met.cap_scale * cap_mult)
        floor_s = floor_h / ext_h
        if floor_frac > 0.0:
            floor_s = max(floor_s, floor_frac * self.met.short_side / max(ext_w, ext_h))
        label_floor = 0.0
        if role_for_floor is not None:
            floor_s = max(floor_s, self._text_floor(unit, role_for_floor))
            label_floor = self.met.min_font(role_for_floor)
        it = _Scaled(unit, self.elements, ideal_s, floor_s, self.prio(prio), w_frac, self.m, label_floor,
                     tier=tier_of(prio))
        it.cap_s = self.met.cap_scale * cap_mult
        if self.rescue:
            _pre_relax(it)
        return it

    def images(self, H: float) -> _Item | None:
        """v11 (UNITS): the secondary pictures as one column item, sized from
        their own share of the source (not a chip's): one picture alone, or a
        row of them side by side (a gallery strip)."""
        if not self.slots.images:
            return None
        src = self.analysis.canvas
        items: list[_Item] = []
        for u in self.slots.images:
            share = u.box.h / src.h
            ideal = clamp(share, 0.12, 0.35) * H
            items.append(self.scaled(u, ideal, 0.06 * H, PRIO_HERO, 0.85, floor_frac=STAMP_FLOOR_FRAC))
        if len(items) == 1:
            return items[0]
        row = _Row(items, max(3.0, self.met.gap * 0.5))
        row.w_frac = 0.9
        return row

    def logo(self, ideal_h: float, floor_h: float, w_frac: float) -> _Scaled | None:
        if self.slots.logo is None:
            return None
        return self.scaled(self.slots.logo, ideal_h, floor_h, PRIO_PRIMARY, w_frac, cap_mult=1.2,
                           floor_frac=LOGO_FLOOR_FRAC)

    def hero(self, ideal_h: float, floor_h: float, w_frac: float) -> _Scaled | None:
        if self.slots.hero is None:
            return None
        src, t = self.analysis.canvas, self.target
        cap = max(t.w / src.w, t.h / src.h) / self.met.cap_scale   # never beyond the larger axis scale
        return self.scaled(self.slots.hero, ideal_h, floor_h, PRIO_HERO, w_frac, cap_mult=cap,
                           floor_frac=HERO_FLOOR_FRAC)

    def cta(self, ideal_h: float, w_frac: float, natural_width: bool = False) -> _Item | None:
        u = self.slots.cta
        if u is None:
            return None
        if u.primary_type(self.elements) == "text":
            return self.text(u, PRIO_PRIMARY, natural_width)
        return self.scaled(u, ideal_h, 0.0, PRIO_PRIMARY, w_frac, role_for_floor=CTA, cap_mult=1.15)

    def chips(self, badge_h: float, image_h: float, direction: str, w_frac: float = 1.0,
              with_images: bool = True) -> _Chips | None:
        units = list(self.slots.chips) + (list(self.slots.images) if with_images else [])
        if not units:
            return None
        items = []
        for u in units:
            # A sibling row re-flows: its parts are separate chips that the
            # cluster packs into rows / columns of the available size.
            for part in (u.parts if u.is_row else [u]):
                has_text = any(self.elements[e]["type"] == "text" for e in part.ids)
                ideal = badge_h if has_text else image_h
                # A sticker that was big in the source (a date block at 17 % of
                # the poster) keeps some of that share instead of shrinking to a pill.
                share = part.box.h / self.analysis.canvas.h
                ideal = max(ideal, min(CHIP_MAX_SHARE * self.target.h, share * self.target.h))
                floor_h = 6.0 if has_text else min(10.0, image_h)
                it = self.scaled(part, ideal, floor_h, PRIO_MINOR, 1.0,
                                 role_for_floor=BADGE if has_text else None,
                                 floor_frac=STAMP_FLOOR_FRAC)
                it.anchor_unit = u      # corner pinning follows the row, not the part
                items.append(it)
        cluster = _Chips(items, direction, max(3.0, self.met.gap * 0.5), w_frac)
        cluster.prio = self.prio(PRIO_MINOR)
        return cluster


# ---------------------------------------------------------------------------
# column plans (tall, square, and the text side of wide)


def _pre_relax(item: _Item) -> None:
    """Rescue first (PlanOptions.rescue, v9): the item's floor is lowered the
    way the first two RESCUE_STEPS would lower it, before the k search runs."""
    for tier, factor in RESCUE_STEPS[:2]:
        item.relax(factor, tier)


def _column_fits(items: list[_Item], gaps: list[float], aw: float, ah: float) -> Callable[[float], bool]:
    def fits(k: float) -> bool:
        total = sum(it.size(k, aw if it.applies_w_frac else aw * it.w_frac, ah)[1] for it in items) + sum(gaps)
        return total <= ah + 0.5
    return fits


def _anchored_bottom(item: _Item) -> bool:
    """Did this item's unit hug the bottom edge in the source (a CTA usually does)?"""
    units = [it.unit for it in getattr(item, "items", [item]) if hasattr(it, "unit")]
    return any(u.anchor_y == "bottom" for u in units)


def _place_column(items: list[_Item], gaps: list[float], x0: float, y0: float, aw: float, ah: float,
                  align: str, met: Metrics, out: Placement, top_anchor: bool, bottom_anchor: bool) -> None:
    n = len(items)
    total = sum(it.h for it in items) + sum(gaps)
    leftover = max(0.0, ah - total)
    extra = min(leftover / (len(gaps) + 2), met.gap) if gaps else 0.0
    leftover -= extra * len(gaps)
    before = [0.0] * n
    if n == 1:
        before[0] = 0.0 if top_anchor else (leftover if bottom_anchor else leftover / 2.0)
    elif top_anchor and bottom_anchor:
        if n == 2:
            before[1] = leftover
        else:
            before[1] += leftover / 2.0
            before[-1] += leftover / 2.0
    elif top_anchor:
        before[1] = leftover / 2.0
    elif bottom_anchor:
        before[0] = leftover / 2.0
        before[-1] += leftover / 2.0
    else:
        before[0] = leftover / 2.0
    y = y0
    tops: list[float] = []
    bottoms: list[float] = []
    for i, it in enumerate(items):
        y += before[i]
        if i > 0:
            y += gaps[i - 1] + extra
        if align == "left":
            x = x0
        elif align == "right":
            x = x0 + aw - it.w
        else:
            x = x0 + (aw - it.w) / 2.0
        it.place(x, y, it.w, it.h, out)
        tops.append(y)
        y += it.h
        bottoms.append(y)
    # v10: the region each text may slide in (engine/backdrop.slide_texts):
    # the column's width, and the room down to its neighbours' gaps, so a
    # slide never changes the reading order.
    for i, it in enumerate(items):
        if not isinstance(it, _TextItem):
            continue
        lo = bottoms[i - 1] + gaps[i - 1] if i > 0 else y0
        hi = tops[i + 1] - gaps[i] if i + 1 < n else y0 + ah
        out.slots[it.unit.key] = Box(x0, lo, aw, max(hi - lo, it.h))


def _column_items(b: _Builder, met: Metrics, H: float, with_hero: bool, tight: bool,
                  chips_mode: str = "row") -> tuple[list[_Item], list[float], _Chips | None]:
    """Items of the centred column and the gaps between them. The third value
    is the chips cluster when `chips_mode` is "corner": it is not part of the
    column and the caller pins it to the canvas corners after the k search."""
    items: list[_Item] = []
    text_gap = b.text_gap()
    # The logo keeps the share of the height the designer gave it (within
    # reason); a logo that was 10 % of the poster stays 10 % of the story.
    logo_frac = 0.07
    if b.slots.logo is not None:
        logo_frac = clamp(b.slots.logo.box.h / b.analysis.canvas.h, 0.07, 0.14)
    logo = b.logo(ideal_h=logo_frac * H, floor_h=max(12.0, 0.035 * H), w_frac=0.5)
    if logo:
        items.append(logo)
    texts = b.text_items()
    for t in texts:
        t.w_frac = 0.86 if tight else 0.9
        if t.vertical:
            t.len_share = VERTICAL_LEN_SHARE
    hero_item: _Item | None = None
    if with_hero:
        hero = b.hero(ideal_h=(0.36 if tight else 0.40) * H, floor_h=0.15 * H, w_frac=0.85)
        if hero:
            hero_item = hero
            # Vertical texts that ran along the hero's sides stay there (v5).
            rails = [t for t in texts if t.vertical and _beside(t, b.slots.hero)]
            if rails:
                texts = [t for t in texts if t not in rails]
                hero_item = _HeroRails(hero, rails, max(3.0, met.gap * 0.5), b.slots.hero.box)
                hero_item.w_frac = 0.9
    # The hero keeps its place in the reading order (v5): a poster whose
    # headline sat above the photo keeps the headline above it, and the
    # lines that followed the photo follow it again.
    head = next((t for t in texts if t.unit.role == HEADLINE), None)
    hero_after = (hero_item is not None and head is not None
                  and b.slots.hero.reading_index > head.unit.reading_index)
    if hero_item is not None and not hero_after:
        items.append(hero_item)
        items.extend(texts)
    elif hero_item is not None:
        hero_idx = b.slots.hero.reading_index
        items.extend(t for t in texts if t.unit.reading_index < hero_idx)
        items.append(hero_item)
        items.extend(t for t in texts if t.unit.reading_index >= hero_idx)
    else:
        items.extend(texts)
    gallery = b.images(H)
    if gallery is not None:
        # beside the hero in the reading order: right after it, or, with no
        # hero in the column, before the first text that followed it
        if hero_item is not None and hero_item in items:
            items.insert(items.index(hero_item) + 1, gallery)
        else:
            idx = min(u.reading_index for u in b.slots.images)
            pos = next((i for i, it in enumerate(items)
                        if isinstance(it, _TextItem) and it.unit.reading_index > idx), len(items))
            items.insert(pos, gallery)
    chips = b.chips(badge_h=0.045 * H, image_h=0.07 * H, direction="row", w_frac=0.95, with_images=False)
    cta = b.cta(ideal_h=0.065 * H, w_frac=0.7)
    if cta:
        cta.w_frac = 0.7 if isinstance(cta, _Scaled) else 0.9
    corner_chips: _Chips | None = None
    if chips and chips_mode == "beside_cta" and cta:
        chips.w_frac = 1.0
        cta.w_frac = 1.0
        if isinstance(cta, _TextItem):
            cta.natural_width = True
        chips_first = _mean_reading_index(b.slots.chips) < b.slots.cta.reading_index
        pair = [chips, cta] if chips_first else [cta, chips]
        shares = [0.45, 0.55] if chips_first else [0.55, 0.45]
        row = _Row(pair, met.gap, shares)
        row.w_frac = 0.95
        items.append(row)
    else:
        if chips and chips_mode == "corner":
            corner_chips = chips
        elif chips:
            items.append(chips)
        if cta:
            items.append(cta)
    fine = b.fine_items()
    for f in fine:
        f.w_frac = 0.92
    items.extend(fine)

    gaps = []
    for a, c in zip(items, items[1:]):
        if isinstance(a, _TextItem) and isinstance(c, _TextItem) and a in texts and c in texts:
            gaps.append(text_gap(texts[0].cap))
        else:
            gaps.append(met.gap)
    return items, gaps, corner_chips


def _mean_reading_index(units: list[Unit]) -> float:
    idx = [u.reading_index for u in units if u.reading_index >= 0]
    return sum(idx) / len(idx) if idx else 1e9


def _absorb_leftover(items: list[_Item], gaps: list[float], aw: float, ah: float, met: Metrics) -> None:
    """v5: when the column is shorter than its slot once everything is at
    its ideal (a tall strip whose text is bound by the width), the hero
    grows into the room, up to HERO_MAX_SHARE of the canvas height and its
    own ceiling, instead of leaving the strip empty."""
    hero = next((it for it in items if it.tier == TIER_HERO and hasattr(it, "grow")), None)
    if hero is None:
        return
    total = sum(it.h for it in items) + sum(gaps)
    leftover = ah - total
    if leftover <= met.gap:
        return
    max_h = min(hero.h + leftover - met.gap, HERO_MAX_SHARE * met.target.h)
    hero.grow(max_h, aw * (hero.w_frac if hero.applies_w_frac else 1.0))   # type: ignore[attr-defined]


def _place_corner_chips(chips: _Chips, k: float, met: Metrics, area: Box, out: Placement) -> None:
    """Pin each chip to the corner of `area` it came from in the source
    (its anchors), stacking chips that share a corner. Sizes come from the
    column's shrink factor so the stickers match the rest of the layout."""
    g = chips.gap
    cursors: dict[tuple[str, str], float] = {}
    row_seen: dict[str, int] = {}
    for c in chips.chips:
        c.size(k, area.w * 0.45, area.h * 0.3)
        u = getattr(c, "anchor_unit", None) or c.unit
        src = u.box
        if u.anchor_x in ("left", "right"):
            hx = u.anchor_x
        else:
            hx = "left" if src.cx <= met.target.w / 2.0 else "right"
        if u is not c.unit:
            # a sibling row spreads over both corners of its edge, left to right
            n = row_seen.get(u.key, 0)
            row_seen[u.key] = n + 1
            hx = ("left", "right")[n % 2] if u.anchor_x == "center" or u.anchor_x == "fluid" else (hx if n % 2 == 0 else ("right" if hx == "left" else "left"))
        if u.anchor_y in ("top", "bottom"):
            vy = u.anchor_y
        else:
            vy = "top" if src.cy <= met.target.h / 2.0 else "bottom"
        used = cursors.get((hx, vy), 0.0)
        x = area.x if hx == "left" else area.right - c.w
        y = area.y + used if vy == "top" else area.bottom - used - c.h
        c.place(x, y, c.w, c.h, out)
        cursors[(hx, vy)] = used + c.h + g


def _plan_column(b: _Builder, met: Metrics, out: Placement, family: str,
                 opts: PlanOptions = DEFAULT_OPTIONS) -> None:
    t = met.target
    m = met.margin
    aw, ah = t.w - 2 * m, t.h - 2 * m
    items, gaps, corner_chips = _column_items(b, met, t.h, with_hero=True,
                                              tight=(family == SQUARE), chips_mode=opts.chips)
    k = 1.0
    if items:
        ah_fit = ah * (1.0 - AIR_SHARE) if opts.air else ah
        k, _ = _largest_k(_column_fits(items, gaps, aw, ah_fit), items + ([corner_chips] if corner_chips else []))
        _absorb_leftover(items, gaps, aw, ah_fit, met)
        top = bool(b.slots.logo) and isinstance(items[0], _Scaled) and items[0].unit is b.slots.logo
        bottom = bool(b.slots.fine) or _anchored_bottom(items[-1])
        _place_column(items, gaps, m, m, aw, ah, "center", met, out, top, bottom)
    if corner_chips is not None:
        _place_corner_chips(corner_chips, k, met, Box(m, m, aw, ah), out)


def _plan_wide(b: _Builder, met: Metrics, out: Placement, opts: PlanOptions = DEFAULT_OPTIONS) -> None:
    t = met.target
    m = met.margin
    aw, ah = t.w - 2 * m, t.h - 2 * m
    # v11 "bleed": the hero runs flush with its side edge at the canvas's full
    # height (the designer's photo-off-the-edge composition); the column keeps
    # the margins and a wider gutter.
    bleed = opts.compose == "bleed"
    hero_h = t.h if bleed else ah
    hero = b.hero(ideal_h=hero_h, floor_h=0.4 * ah, w_frac=1.0)
    items, gaps, corner_chips = _column_items(b, met, t.h, with_hero=False, tight=False,
                                              chips_mode=opts.chips)
    side = opts.hero_side if opts.hero_side in ("left", "right") else b.slots.hero_side
    align = b.slots.text_align if hero else "center"
    col_frac = 1.0 if hero else 0.8

    share = HERO_SHARES[0]

    def col_w() -> float:
        if hero is None:
            return aw * col_frac
        # The hero stands beside the column at its full size (the height and
        # its share of the width bound it); only the column shrinks with k.
        # v3 shrank the hero with the column's k although it never competed
        # for the column's height.
        if bleed:
            hero.size(1.0, t.w * share, hero_h)
            return t.w - hero.w - m - max(met.gap, m)
        hero.size(1.0, aw * share, ah)
        return aw - hero.w - met.gap

    head = next((t for t in items if isinstance(t, _TextItem) and t.unit.role == HEADLINE
                 and not t.rotation), None)

    def head_lines() -> int:
        return head.m.layout(head.el, head.font, max(1.0, head.own_w)).n_lines if head is not None else 0

    def fits(k: float) -> bool:
        cw = col_w()
        if not items:
            return True
        if not _column_fits(items, gaps, cw, ah * (1.0 - AIR_SHARE) if opts.air else ah)(k):
            return False
        # A headline on more than HEAD_MAX_LINES lines is a defect to the
        # judge however well it fits (v4): the search shrinks it (and the
        # column's narrower share of the width is tried) before that.
        return k <= 0.0 or head_lines() <= HEAD_MAX_LINES

    # The hero takes the biggest share of the width that leaves the column
    # most of its size (v4): a square product shot at 45 % of a 16:9 canvas
    # squeezed the headline onto four lines in v3. Among the shares tried,
    # the first whose column keeps >= HERO_SHARE_K_TOL of the best k wins.
    k = 1.0
    if hero is not None and items:
        tried: list[tuple[float, float]] = []
        for sh in HERO_SHARES:
            share = sh
            k, _ = _largest_k(fits, items + ([corner_chips] if corner_chips else []))
            tried.append((sh, k))
            if k >= 1.0:
                break        # the biggest hero already leaves the column at full size
        best_k = max(k for _, k in tried)
        share = next(sh for sh, k in tried if k >= HERO_SHARE_K_TOL * best_k)
        k, _ = _largest_k(fits, items + ([corner_chips] if corner_chips else []))
    else:
        k, _ = _largest_k(fits, items + ([corner_chips] if corner_chips else []))
        col_w()
    cw = aw - hero.w - met.gap if hero is not None else aw * col_frac
    if hero is not None and bleed:
        cw = t.w - hero.w - m - max(met.gap, m)
        hx = 0.0 if side == "left" else t.w - hero.w
        hero.place(hx, (t.h - hero.h) / 2.0, hero.w, hero.h, out)
        x0 = hero.w + max(met.gap, m) if side == "left" else m
    elif hero is not None:
        hx = m if side == "left" else t.w - m - hero.w
        hero.place(hx, m + (ah - hero.h) / 2.0, hero.w, hero.h, out)
        x0 = m + hero.w + met.gap if side == "left" else m
    else:
        x0 = m + (aw - cw) / 2.0
    if items:
        top = bool(b.slots.logo) and isinstance(items[0], _Scaled) and items[0].unit is b.slots.logo
        bottom = bool(b.slots.fine) or _anchored_bottom(items[-1])
        _place_column(items, gaps, x0, m, cw, ah, align, met, out, top, bottom)
    if corner_chips is not None:
        _place_corner_chips(corner_chips, k, met, Box(x0, m, cw, ah), out)


def _plan_header(b: _Builder, met: Metrics, out: Placement, opts: PlanOptions = DEFAULT_OPTIONS) -> bool:
    """v11 "header" composition (wide): the headline runs across the top of
    the canvas on at most two lines; below it the hero stands beside a column
    with everything else (logo, copy, chips, CTA, fine print). False when the
    design has no headline or no hero (the caller falls back to the wide plan)."""
    t = met.target
    m = met.margin
    aw, ah = t.w - 2 * m, t.h - 2 * m
    if b.slots.hero is None:
        return False
    items, gaps, _ = _column_items(b, met, t.h, with_hero=False, tight=False, chips_mode="row")
    head = next((it for it in items if isinstance(it, _TextItem) and it.unit.role == HEADLINE
                 and not it.rotation), None)
    if head is None:
        return False
    i = items.index(head)
    rest = items[:i] + items[i + 1:]
    rest_gaps = [met.gap] * max(0, len(rest) - 1)
    for j in range(len(rest) - 1):
        # keep the column's own gaps where both neighbours were adjacent before
        a, c = rest[j], rest[j + 1]
        if a in items and c in items and items.index(c) == items.index(a) + 1:
            rest_gaps[j] = gaps[items.index(a)]
    head.w_frac = 1.0
    hero = b.hero(ideal_h=0.62 * ah, floor_h=0.25 * ah, w_frac=1.0)
    side = opts.hero_side if opts.hero_side in ("left", "right") else b.slots.hero_side
    align = b.slots.text_align
    lower_items: list[_Item] = [hero]
    shares = [0.45]
    if rest:
        col = _Stack(rest, rest_gaps, align=align)
        lower_items = [hero, col] if side == "left" else [col, hero]
        shares = [0.45, 0.55] if side == "left" else [0.55, 0.45]
    lower = _Row(lower_items, max(met.gap, m), shares)
    whole = _Stack([head, lower], [met.gap * 1.2], align=align)

    def fits(k: float) -> bool:
        w, h = whole.size(k, aw, ah)
        if h > ah + 0.5 or w > aw + 0.5:
            return False
        return k <= 0.0 or head.m.layout(head.el, head.font, max(1.0, head.own_w)).n_lines <= 2

    _largest_k(fits, [whole])
    whole.place(m, m, aw, ah, out)
    return True


# ---------------------------------------------------------------------------
# banner plan (single row)


def _plan_banner(b: _Builder, met: Metrics, out: Placement, opts: PlanOptions = DEFAULT_OPTIONS) -> None:
    t = met.target
    m = met.margin
    aw, ah = t.w - 2 * m, t.h - 2 * m
    gap = met.gap
    compose = opts.compose

    logo = b.logo(ideal_h=0.4 * ah, floor_h=max(10.0, 0.2 * ah), w_frac=0.3)
    # v11 "bleed": the hero fills the strip's full height flush with one end;
    # the row is laid out on the rest of the width.
    bleed = compose == "bleed" and b.slots.hero is not None
    hero = b.hero(ideal_h=t.h if bleed else ah, floor_h=0.35 * ah, w_frac=0.3)
    x_lo = m
    if bleed and hero is not None:
        hero.size(1.0, 0.3 * t.w, t.h)
        aw = t.w - hero.w - m - max(gap, m)
        if opts.hero_side != "right":
            x_lo = hero.w + max(gap, m)
    texts = b.text_items() + b.fine_items()
    column: _TextColumn | None = None     # the column the line policy steers (holds the headline)
    rest_column: _TextColumn | None = None
    chips = b.chips(badge_h=0.26 * ah, image_h=0.3 * ah,
                    direction="col" if opts.chips == "stack" else "auto", w_frac=0.3)
    cta = b.cta(ideal_h=0.38 * ah, w_frac=0.3, natural_width=True)
    tiered: _Stack | None = None
    if texts:
        head = next((x for x in texts if x.unit.role == HEADLINE), texts[0])
        others = [x for x in texts if x is not head]
        if compose == "twotier" and (others or chips or cta):
            # v11: the headline on one line across the strip's text area, the
            # copy, badges and CTA in one row underneath (the designer's
            # two-tier banner); chips and CTA leave the main row.
            column = _TextColumn([head], head, b.text_gap(), w_frac=1.0, max_lines=1, flow=False)
            lower: list[_Item] = []
            shares: list[float] = []
            if others:
                rest_column = _TextColumn(others, others[0], b.text_gap(), w_frac=1.0, max_lines=2)
                lower.append(rest_column)
                shares.append(0.5)
            if chips:
                chips.w_frac = 1.0
                lower.append(chips)
                shares.append(0.25)
            if cta:
                lower.append(cta)
                shares.append(0.25)
            tot = sum(shares)
            row = _Row(lower, gap, [x / tot for x in shares])
            used = (0.0 if bleed or hero is None else 0.3) + (0.15 if logo is not None else 0.0)
            tiered = _Stack([column, row], [max(3.0, 0.5 * gap)], align="left", w_frac=1.0 - used)
            chips = cta = None
            rest_column = None
        elif opts.text_flow == "inline" and others:
            column = _TextColumn([head], head, b.text_gap(), w_frac=0.36)
            rest_column = _TextColumn(others, others[0], b.text_gap(), w_frac=0.3, max_lines=3)
        else:
            column = _TextColumn(texts, head, b.text_gap(), w_frac=0.6)
    body: _Item | None = tiered if tiered is not None else column
    row_hero = None if bleed else hero

    if compose == "bookend" or opts.hero_side != "right":
        order = (logo, row_hero, body, rest_column, chips, cta)
    else:
        order = (logo, body, rest_column, chips, cta, row_hero)
    if compose == "bookend" and opts.hero_side == "right":
        order = (logo, body, rest_column, chips, row_hero, cta)
    items: list[_Item] = [it for it in order if it is not None]
    if bleed and hero is not None:
        hx = 0.0 if opts.hero_side != "right" else t.w - hero.w
        hero.place(hx, (t.h - hero.h) / 2.0, hero.w, hero.h, out)
    if not items:
        return

    def fits(k: float) -> bool:
        # Size *every* item before judging: a plan that cannot fit (twelve
        # lines of copy in a 90 px row) must still leave each item with a
        # real size, or the unsized ones come out as 0x0 boxes, which the
        # official renderer rejects outright.
        total = 0.0
        ok = True
        for it in items:
            w, h = it.size(k, aw, ah)
            if h > ah + 0.5:
                ok = False
            total += w
        return ok and total + gap * (len(items) - 1) <= aw + 0.5

    # Fewest headline lines first, but only while the row is still roomy
    # (k above the threshold); a 1-line headline at half size loses to a
    # 2-line one at full size.
    if column is None:
        k, ok = _largest_k(fits)
    else:
        chosen = None
        for max_lines, min_k in BANNER_LINE_POLICY:
            column.max_lines = max_lines
            k, ok = _largest_k(fits)
            if ok:
                chosen = max_lines
                if k >= min_k:
                    break
        column.max_lines = chosen if chosen is not None else BANNER_LINE_POLICY[-1][0]
        k, ok = _largest_k(fits)
    if not ok:
        k, ok = _largest_k(fits, items)      # rescue mode: lower the minor floors

    leftover = aw - sum(it.w for it in items) - gap * (len(items) - 1)
    if column is not None and column in items and leftover > 0:
        comfy = column.comfortable_width(k, aw * column.w_frac)
        column.widen(k, min(comfy, column.w + leftover), ah)
        leftover = aw - sum(it.w for it in items) - gap * (len(items) - 1)
    leftover = max(0.0, leftover)

    n = len(items)
    max_gap = max(MAX_GAP_MULT * gap, MAX_GAP_FRAC * t.w)
    ends = [it for it in (items[0], items[-1]) if it is logo or it is cta] if n >= 3 else []
    if compose == "bookend" and ends:
        # v11 "bookend": the logo and the CTA hold the two ends of the strip,
        # the rest is centred between them with the cohesion cap on its gaps.
        first = items[0] if items[0] in ends else None
        last = items[-1] if items[-1] in ends else None
        mid = [it for it in items if it is not first and it is not last]
        lo_x = x_lo
        hi_x = x_lo + aw
        if first is not None:
            first.place(lo_x, m + (ah - first.h) / 2.0, first.w, first.h, out)
            lo_x += first.w + gap
        if last is not None:
            last.place(hi_x - last.w, m + (ah - last.h) / 2.0, last.w, last.h, out)
            hi_x -= last.w + gap
        span = hi_x - lo_x
        used = sum(it.w for it in mid) + gap * max(0, len(mid) - 1)
        free = max(0.0, span - used)
        extra = min(free / (len(mid) + 1), max(0.0, max_gap - gap)) if mid else 0.0
        x = lo_x + (free - extra * max(0, len(mid) - 1)) / 2.0
        for it in mid:
            it.place(x, m + (ah - it.h) / 2.0, it.w, it.h, out)
            x += it.w + gap + extra
        return

    # Cohesion (v5): gaps grow only up to a cap; the rest of the leftover
    # centres the row instead of pushing its ends to the canvas edges.
    extra = leftover / (n - 1) if n > 1 else 0.0
    extra = min(extra, max(0.0, max_gap - gap))
    x = x_lo + (leftover - extra * (n - 1)) / 2.0
    for it in items:
        it.place(x, m + (ah - it.h) / 2.0, it.w, it.h, out)
        x += it.w + gap + extra


# ---------------------------------------------------------------------------
# entry point


def floorplan_relayout(analysis: Analysis, target: Box, elements: dict[str, dict[str, Any]],
                       measurer: TextMeasurer | None = None,
                       options: PlanOptions | None = None) -> Placement:
    """Lay the analysed design out on `target` with the floor plan of its
    ratio family. `elements` maps element id -> source element (needed for
    text measurement and rotations). `options` picks one of the plan's
    variants (defaults reproduce the single v1 layout)."""
    m = measurer or default_measurer()
    opts = options or DEFAULT_OPTIONS
    family = ratio_family(target)
    met = Metrics.for_canvas(analysis.canvas, target, family, opts.spacing, opts.scale_bias)
    src = analysis.canvas
    s = min(target.w / src.w, target.h / src.h)
    out = Placement()

    deco = decorative_boxes(analysis, target, s)
    for unit in analysis.units:
        if unit.role == BACKGROUND:
            for eid, mb in unit.members.items():
                out.boxes[eid] = _cover_box(mb, src, target)
                out.font_scale[eid] = s
        elif unit.role == DECORATIVE:
            _place_members(unit, deco.get(unit.key) or _map_decorative(unit.box, src, target, s), s, out)
            for eid in unit.ids:
                out.font_scale[eid] = s

    b = _Builder(analysis, elements, met, m, family, uniform=opts.fill, rescue=opts.rescue)
    plan = family
    if family == SQUARE and opts.hero_side in ("left", "right") and b.slots.hero is not None:
        plan = WIDE          # hero on one side, column on the other, on a square
    elif family == WIDE and opts.hero_side == "stack":
        plan = TALL          # one centred column on a wide canvas
    elif family == BANNER and opts.text_flow == "column" and target.h >= BANNER_COLUMN_MIN_H:
        plan = WIDE          # hero beside one column holding everything else (v4)
    if plan in (TALL, SQUARE):
        _plan_column(b, met, out, family, opts)
    elif plan == WIDE:
        if not (opts.compose == "header" and _plan_header(b, met, out, opts)):
            _plan_wide(b, met, out, opts)
    else:
        _plan_banner(b, met, out, opts)

    if b.strip:
        _keep_inside(analysis, out, target)

    # Panels (colour zones) are rebuilt around the content they carried.
    place_panels(analysis, out, target, mode="wrap", measurer=m)
    # v10: texts slide inside their slots to the backdrop that looks like
    # their source backdrop (engine/backdrop.py); panels re-wrap after.
    if slide_backdrop(analysis, out, target, elements, m):
        place_panels(analysis, out, target, mode="wrap", measurer=m)

    # Safety net: anything a plan did not place keeps its proportions and
    # its relative position (should not happen; keeps every id present).
    for unit in analysis.units:
        if all(eid in out.boxes for eid in unit.ids):
            continue
        cx, cy = unit.box.cx / src.w * target.w, unit.box.cy / src.h * target.h
        new = Box(cx - unit.box.w * s / 2, cy - unit.box.h * s / 2, unit.box.w * s, unit.box.h * s)
        _place_members(unit, new, s, out)
        for eid in unit.ids:
            out.font_scale[eid] = s
    return out


STRIP_EDGE = 0.015          # the rectifier's on-canvas margin (rectify.MARGIN_FRAC of the short side)


def _keep_inside(analysis: Analysis, out: Placement, target: Box) -> None:
    """v11 strip composer: a content unit the plan left poking past the
    canvas edge by a few pixels (a 146 px headline centred on a column whose
    footprint was wider than the text) slides back inside, so the rectifier
    does not shrink it under its floor to make it fit."""
    mg = max(1.0, STRIP_EDGE * min(target.w, target.h))
    for unit in analysis.content_units():
        boxes = [out.boxes[e] for e in unit.ids if e in out.boxes]
        if not boxes:
            continue
        hull = Box.union(boxes)
        dx = dy = 0.0
        if hull.w <= target.w - 2 * mg:
            if hull.x < mg:
                dx = mg - hull.x
            elif hull.right > target.w - mg:
                dx = target.w - mg - hull.right
        if hull.h <= target.h - 2 * mg:
            if hull.y < mg:
                dy = mg - hull.y
            elif hull.bottom > target.h - mg:
                dy = target.h - mg - hull.bottom
        if dx or dy:
            for e in unit.ids:
                if e in out.boxes:
                    bx = out.boxes[e]
                    out.boxes[e] = Box(bx.x + dx, bx.y + dy, bx.w, bx.h)


def slide_backdrop(analysis: Analysis, out: Placement, target: Box, elements: dict[str, dict[str, Any]],
                   m: TextMeasurer) -> bool:
    """v10: `engine.backdrop.slide_texts` on a placement whose column placer
    recorded slots; records the moves in `out.notes["slide"]`. Off (and a
    no-op) unless CHAMELEON_BACKGROUND / CHAMELEON_BACKDROP_SLIDE are on."""
    from engine import backdrop
    if not out.slots or not backdrop.enabled("slide"):
        return False
    try:
        moved = backdrop.slide_texts(analysis, out, target, elements, m)
    except Exception:  # pragma: no cover - the slide is a bonus, never a dependency
        import logging
        logging.getLogger(__name__).exception("backdrop slide failed; placement kept")
        return False
    if moved:
        out.notes["slide"] = moved
    return bool(moved)
