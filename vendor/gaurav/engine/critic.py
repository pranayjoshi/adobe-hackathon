"""Critic loop (v9): defect-driven regeneration behind CHAMELEON_CRITIC.

The judge loop (engine/resize.py) generates a fixed set of candidates and
keeps the best. When every candidate shares a failure (the headline below
the readable floor on a 160 px strip, a headline whose contrast collapsed
because the plate it sat on stayed behind, two units colliding in every
plan) the judge picks the least bad instead of fixing the cause. The critic
reads the judged candidates, names the shared failure (`diagnose`) and
turns it into *repair recipes*: concrete changes to the generation knobs
(PlanOptions / StructureOptions overrides, an anchor content scale) and
*directives* that `engine.resize.realize` applies to the materialized
elements before the rectifier runs (`apply_directives`): raise a text to
its floor and re-wrap it, bring its backing plate along, lay it on the hero
it was layered on in the source, pin a unit back to the edge it hugged,
shrink the less important half of a colliding pair. `engine.candidates.
generate_repairs` builds the new candidates from the recipes; the judge
scores them with the rest, so the critic never decides, it only proposes.

Everything here is keyed on roles and relations from the analysis, never
on element ids or canvas sizes; recipes are general rules.

CHAMELEON_CRITIC=0 switches the loop off (byte-identical output);
CHAMELEON_CRITIC_ROUNDS and CHAMELEON_CRITIC_TARGET override the budget.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from engine import backdrop
from engine.analyze import (
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
    SUBHEAD,
    Analysis,
    Unit,
    _rgb,
)
from engine.floorplan import Metrics, _min_width_for_lines, ratio_family
from engine.geometry import Box, clamp
from engine.rectify import _on_fraction, overlap_rules, rotated_extent, visible_box
from engine.relayout import CONTENT_MARGIN_FRAC, _clip, _edges_touched, _is_rect, _stick_edges, stack_boxes
from engine.score import CONTRAST_GOOD, contrast_ratio
from engine.textfit import TextMeasurer, apply_font_size, default_measurer, scaled_letter_spacing

DEFAULT_ON = "1"                 # decided by the numbers in docs/CRITIC_LOOP.md
ROUNDS = 2                       # CHAMELEON_CRITIC_ROUNDS
TARGET = 88.0                    # CHAMELEON_CRITIC_TARGET: stop when the best clean candidate reaches this
                                 # (60th percentile of the clean winners under the v8 judge, eval run v8)
MAX_REPAIRS = 6                  # new candidates per round
SHARE_MIN = 0.5                  # a hard fail is "shared" when at least this share of the candidates has it ...
TOP_SOFT = 3                     # ... a soft weakness when the best this many clean candidates all show it
SEEDS = 2                        # seeds (best candidates) a directive recipe is applied to
BACKING_MIN_ON = 0.5             # a plate / panel is a text's backing when this much of its ink lay on it in the source
BACKING_MIN_CONTRAST = 3.0       # ... and the text's colour reads against the plate's fill at this ratio
LAYERED_MIN_ON = 0.3             # a text layered on the hero in the source (rectify.LAYER_SRC_MIN)
HEAD_MAX_LINES = 3               # a raised headline wraps on at most this many lines, a CTA label on one
PAD_MIN_EM = 0.3                 # a backing plate keeps at least this x the font around the text
SHRINK_COLLIDER = 0.8            # the less important half of a colliding pair starts this much smaller
PIN_ROLES = (LOGO, BADGE, FINE_PRINT, CTA)   # units the "pin to source edges" recipe may move back to an edge

# Soft-term weakness: the top clean candidates all score below the 25th
# percentile of the winners of the v7 eval run (eval/data/runs/v7) on a term
# that has a repair mechanism. Terms without a recipe are not diagnosed.
TYPICAL: dict[str, float] = {      # p25 of the clean winners under the v8 judge (eval run v8)
    "faith.anchors": 0.44,
    "read.contrast": 0.45,
    "read.lines": 0.95,            # any headline past 3 lines / CTA past 1 (p25 is 1.0)
    "read.presence": 0.90,
    "layout.size": 0.75,
    "layout.spacing": 0.90,
    "layout.cohesion": 0.95,
    "faith.rows": 0.95,            # v8 row integrity: 99 % of winners score 1.0
}

_RX_FLOOR = re.compile(r"^(\S+) (headline|cta) font ([\d.]+) below readable ([\d.]+)$")
_RX_CONTRAST = re.compile(r"^(\S+) unreadable contrast ([\d.]+) \(source (.+)\)$")
_RX_STRADDLE = re.compile(r"^(\S+) straddles an edge")
_RX_COLLIDE = re.compile(r"^(\S+) collides with (\S+) \(")
_RX_OFF = re.compile(r"^(\S+) off canvas$")
_RX_OVERFLOW = re.compile(r"^(\S+) text overflows its box$")


# ---------------------------------------------------------------------------
# settings


def enabled() -> bool:
    return os.environ.get("CHAMELEON_CRITIC", DEFAULT_ON).strip().lower() not in ("0", "false", "no", "off")


def settings() -> tuple[int, float]:
    """(rounds, target) from the environment, read per call so tests can flip them."""
    try:
        rounds = int(os.environ.get("CHAMELEON_CRITIC_ROUNDS", "") or ROUNDS)
    except ValueError:
        rounds = ROUNDS
    try:
        target = float(os.environ.get("CHAMELEON_CRITIC_TARGET", "") or TARGET)
    except ValueError:
        target = TARGET
    return max(0, rounds), target


# ---------------------------------------------------------------------------
# data


@dataclass
class Recipe:
    """One concrete repair: plan knob overrides and / or realize-time directives."""

    label: str                                   # short, shown in the candidate name ("+fix:headline to floor")
    plan: dict[str, Any] = field(default_factory=dict)        # PlanOptions overrides (floor-plan seeds)
    structure: dict[str, Any] = field(default_factory=dict)   # StructureOptions overrides (structure seeds)
    directives: dict[str, Any] = field(default_factory=dict)  # see apply_directives

    @property
    def key(self) -> str:
        return repr((sorted(self.plan.items()), sorted(self.structure.items()), _freeze(self.directives)))


@dataclass
class Diagnosis:
    rule: str                     # below_floor | contrast_drop | straddle | collision | off_canvas | overflow | weak:<term>
    units: tuple[str, ...]        # element ids / unit keys involved
    detail: str
    shared: int                   # candidates showing it
    of: int                       # candidates examined
    hard: bool
    recipes: list[Recipe] = field(default_factory=list)
    names: frozenset = frozenset()   # candidates showing it (hard: recipes apply to these seeds only)

    def summary(self) -> dict[str, Any]:
        return {"rule": self.rule, "units": list(self.units), "detail": self.detail,
                "shared": f"{self.shared}/{self.of}", "recipes": [r.label for r in self.recipes]}

    def applies_to(self, name: str) -> bool:
        return not self.hard or not self.names or name in self.names


def _freeze(d: Any) -> Any:
    if isinstance(d, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in d.items()))
    if isinstance(d, (list, tuple, set)):
        return tuple(_freeze(v) for v in d)
    return d


# ---------------------------------------------------------------------------
# diagnosis


def diagnose(judged: list[Any], analysis: Analysis, source: dict[str, Any], target: Box,
             measurer: TextMeasurer | None = None) -> list[Diagnosis]:
    """Shared failures of the judged candidates (Judged-like: .name, .fallback,
    .score with .hard / .breakdown / .notes, .design), most severe first.
    Hard fails are diagnosed when no real candidate is clean (what every
    layout gets wrong is the generator's doing); soft weaknesses on the
    best clean candidates otherwise."""
    m = measurer or default_measurer()
    real = [j for j in judged if not getattr(j, "fallback", False)]
    if not real:
        real = list(judged)
    if not real:
        return []
    clean = [j for j in real if not j.score.hard]
    out: list[Diagnosis] = []
    if not clean:
        out.extend(_hard_diagnoses(real, analysis, source, target, m))
    else:
        out.extend(_soft_diagnoses(clean[:TOP_SOFT], analysis, source, target, m))
    return out


def _hard_diagnoses(pool: list[Any], analysis: Analysis, source: dict[str, Any], target: Box,
                    m: TextMeasurer) -> list[Diagnosis]:
    """The best candidate's hard fails (it is what would ship) and every
    kind that at least SHARE_MIN of the pool shows, best's first."""
    counts: dict[tuple, int] = {}
    first: dict[tuple, str] = {}
    names: dict[tuple, set[str]] = {}
    for j in pool:
        seen: set[tuple] = set()
        for h in j.score.hard:
            k = _hard_kind(h)
            if k is None or k in seen:
                continue
            seen.add(k)
            counts[k] = counts.get(k, 0) + 1
            first.setdefault(k, h)
            names.setdefault(k, set()).add(j.name)
    best_kinds = {_hard_kind(h) for h in pool[0].score.hard}
    need = max(1, math.ceil(SHARE_MIN * len(pool)))
    out: list[Diagnosis] = []
    for k, n in sorted(counts.items(), key=lambda kv: (kv[0] not in best_kinds, -kv[1], kv[0])):
        if n < need and k not in best_kinds:
            continue
        rule, units = k[0], k[1:]
        shown = next(j for j in pool if j.name in names[k])
        recipes = _recipes_for_hard(rule, units, analysis, source, target, m, shown)
        if not recipes:
            continue
        out.append(Diagnosis(rule, units, first[k], n, len(pool), True, recipes, frozenset(names[k])))
    return out


def _hard_kind(h: str) -> tuple | None:
    mt = _RX_FLOOR.match(h)
    if mt:
        return ("below_floor", mt.group(1))
    mt = _RX_CONTRAST.match(h)
    if mt:
        return ("contrast_drop", mt.group(1))
    mt = _RX_STRADDLE.match(h)
    if mt:
        return ("straddle", mt.group(1))
    mt = _RX_COLLIDE.match(h)
    if mt:
        a, b = sorted((mt.group(1), mt.group(2)))
        return ("collision", a, b)
    mt = _RX_OFF.match(h)
    if mt:
        return ("off_canvas", mt.group(1))
    mt = _RX_OVERFLOW.match(h)
    if mt:
        return ("overflow", mt.group(1))
    return None


def hard_kind_of(d: Diagnosis) -> tuple:
    """The `_hard_kind` key a hard diagnosis stands for (resize.repair uses it
    to tell which repairs removed what they were built for)."""
    return (d.rule, *d.units)


def _soft_diagnoses(top: list[Any], analysis: Analysis, source: dict[str, Any], target: Box,
                    m: TextMeasurer) -> list[Diagnosis]:
    out: list[Diagnosis] = []
    weak: list[tuple[float, str]] = []
    for term, typical in TYPICAL.items():
        vals = [j.score.breakdown.get(term) for j in top]
        if any(v is None for v in vals):
            continue
        if all(v < typical for v in vals):
            weak.append((min(vals) - typical, term))
    for gap, term in sorted(weak):
        recipes = _recipes_for_weak(term, top[0], analysis, source, target, m)
        if not recipes:
            continue
        units = tuple(sorted({u for r in recipes for u in _recipe_units(r)}))
        out.append(Diagnosis(f"weak:{term}", units, f"{term} {top[0].score.breakdown[term]:.2f} < typical {TYPICAL[term]:.2f}",
                             len(top), len(top), False, recipes))
    return out


def _recipe_units(r: Recipe) -> list[str]:
    units: list[str] = []
    for k, v in r.directives.items():
        if isinstance(v, dict):
            units.extend(v.keys())
        elif isinstance(v, (list, tuple)):
            units.extend(v)
    return units


# ---------------------------------------------------------------------------
# recipes


def _recipes_for_hard(rule: str, units: tuple[str, ...], analysis: Analysis, source: dict[str, Any],
                      target: Box, m: TextMeasurer, best: Any) -> list[Recipe]:
    roles = analysis.roles
    if rule == "below_floor":
        eid = units[0]
        out = [Recipe(f"{roles.get(eid, 'text')} to floor", directives={"floor": [eid], "cap_floor": True})]
        out.append(Recipe("secondary copy yields", plan={"rescue": True}, directives={"floor": [eid], "cap_floor": True}))
        out.append(Recipe("tight, shy plan", plan={"spacing": 0.75, "scale_bias": 0.85},
                          directives={"floor": [eid], "cap_floor": True}))
        return out
    if rule in ("contrast_drop", "straddle"):
        return _contrast_recipes(units[0], analysis, source, best, m) + (
            [Recipe("tighter column", plan={"spacing": 0.75})] if rule == "straddle" else [])
    if rule == "collision":
        a, b = units[0], units[1]
        ua, ub = _unit(analysis, a), _unit(analysis, b)
        victim = a if (ua is not None and ub is not None and ua.importance <= ub.importance) else b
        out = [Recipe("colliders apart", directives={"shrink": {victim: SHRINK_COLLIDER}}, plan={"spacing": 0.75})]
        if any(u is not None and u.role == BADGE for u in (ua, ub)):
            out.append(Recipe("badges to the corners", plan={"chips": "corner"}))
        return out
    if rule == "off_canvas":
        return [Recipe("unit back on canvas", directives={"shrink": {units[0]: 0.85}})]
    if rule == "overflow":
        return [Recipe("text refitted", directives={"refit": [units[0]]})]
    return []


def _recipes_for_weak(term: str, best: Any, analysis: Analysis, source: dict[str, Any], target: Box,
                      m: TextMeasurer) -> list[Recipe]:
    roles = analysis.roles
    design = best.design
    by_id = {el["id"]: el for el in design["elements"]}
    if term == "faith.anchors":
        lost = lost_anchors(analysis, design, target)
        out: list[Recipe] = []
        if lost:
            out.append(Recipe("pin to source edges", directives={"pin": lost}))
        if any(u.role == BADGE for u in analysis.content_units()):
            out.append(Recipe("badges to the corners", plan={"chips": "corner"}))
        return out
    if term == "read.contrast":
        worst = best.score.notes.get("contrast") or {}
        for eid, ratio in sorted(worst.items(), key=lambda kv: kv[1]):
            if ratio >= CONTRAST_GOOD or roles.get(eid) not in (HEADLINE, CTA, SUBHEAD):
                continue
            out = _contrast_recipes(eid, analysis, source, best, m)
            if out:
                return out
        return []
    if term == "read.lines":
        heads = [eid for eid, r in roles.items() if r in (HEADLINE, CTA) and eid in by_id and by_id[eid]["type"] == "text"]
        heads = [eid for eid in heads if m.layout(by_id[eid]).n_lines > (HEAD_MAX_LINES if roles[eid] == HEADLINE else 1)]
        return [Recipe("headline on fewer lines", directives={"widen": heads})] if heads else []
    if term in ("read.presence", "layout.size"):
        return [Recipe("bolder, fill", plan={"scale_bias": 1.3, "fill": True})]
    if term in ("layout.spacing", "layout.cohesion"):
        return [Recipe("tighter plan", plan={"spacing": 0.75})]
    if term == "faith.rows":
        return [Recipe("badges in a row", plan={"chips": "row"})]
    return []


def _contrast_recipes(eid: str, analysis: Analysis, source: dict[str, Any], best: Any,
                      m: TextMeasurer) -> list[Recipe]:
    """What made the text readable in the source, and what covers it now:
    bring its plate along, lay it on the picture it was layered on, or move
    it off a picture the designer never put it on."""
    role = analysis.roles.get(eid, "text")
    out: list[Recipe] = []
    backing = find_backing(eid, analysis, source, m)
    if backing is not None:
        out.append(Recipe(f"{role} on its plate", directives={"backing": {eid: backing}}))
    picture = layered_picture(eid, analysis, m)
    if picture is not None:
        # v10: a `cover` picture is cropped differently in every box, so the
        # pixels under the source position are not the source's; the text
        # goes to the band of the picture where its colour reads instead
        # (engine/backdrop.quiet_position). A `contain` picture keeps the
        # source move and tries the quiet area as a second recipe.
        quiet = backdrop.enabled("quiet")
        if not (quiet and _is_cover(picture, analysis)):
            out.append(Recipe(f"{role} on the picture", directives={"onto": {eid: picture}}))
        if quiet:
            out.append(Recipe(f"{role} onto the quiet area", directives={"quiet": {eid: picture}}))
    cover = covering_unit(eid, analysis, best.design, m)
    if cover is not None and cover != picture and cover != backing:
        out.append(Recipe(f"{role} off the {analysis.unit(cover).role}", directives={"off": {eid: cover}}))
    return out


def _is_cover(key: str, analysis: Analysis) -> bool:
    """Is the unit's primary element a `cover`-fitted image?"""
    u = _unit(analysis, key)
    if u is None:
        return False
    el = analysis.source_elements.get(u.primary or u.ids[0], {})
    return el.get("type") == "image" and str(el.get("fit", "contain") or "contain") == "cover"


# ---------------------------------------------------------------------------
# relations the recipes are keyed on


def _unit(analysis: Analysis, key: str) -> Unit | None:
    for u in analysis.units:
        if u.key == key:
            return u
    k = analysis.element_unit.get(key)
    if k is not None:
        return analysis.unit(k)
    return None


def find_backing(eid: str, analysis: Analysis, source: dict[str, Any], m: TextMeasurer) -> str | None:
    """The flat shape or panel the text's ink lay on in the source (at least
    BACKING_MIN_ON of it) whose fill gives the text's colour a readable
    contrast: the plate that made the text readable there. None when the
    text sat on the canvas, on a picture, or on nothing readable."""
    el = analysis.source_elements.get(eid)
    if el is None or el["type"] != "text":
        return None
    unit_key = analysis.element_unit.get(eid)
    if unit_key is None:
        return None
    overlap_rules(analysis, m)                       # fills analysis.source_visible
    text_unit = analysis.unit(unit_key)
    try:
        ink = [visible_box(el, text_unit.members[eid], None, m)]
    except Exception:
        return None
    rgb = _rgb(el.get("color")) or (0, 0, 0)
    z_text = _z(analysis, eid)
    best: tuple[float, str] | None = None
    for u in analysis.units:
        if u.key == unit_key or u.role in (BACKGROUND, DECORATIVE) or u.role == HERO or u.role == LOGO:
            continue
        if max(_z(analysis, i) for i in u.ids) >= z_text:
            continue          # drawn over the text in the source: it hid it, it did not back it
        shape_el = analysis.source_elements.get(u.ids[0]) if len(u.ids) == 1 or u.kind == "stack" else None
        if shape_el is None or shape_el["type"] != "shape":
            continue
        fill = _rgb(shape_el.get("fill_color"))
        if fill is None:
            continue
        on = _on_fraction(ink, [Box.union(u.members[i] for i in u.ids)])
        if on < BACKING_MIN_ON:
            continue
        if contrast_ratio(rgb, fill) < BACKING_MIN_CONTRAST:
            continue
        if best is None or on > best[0]:
            best = (on, u.key)
    return best[1] if best else None


def _picture_units(analysis: Analysis) -> list[Unit]:
    """Units a text can be read against: the hero, other pictures and
    decorative photos (never the background, which `_cover_box` keeps
    everywhere anyway)."""
    src = analysis.source_elements
    out = []
    for u in analysis.units:
        if u.role == BACKGROUND or not u.ids:
            continue
        if u.role == HERO or (u.role in (IMAGE, DECORATIVE) and src.get(u.ids[0], {}).get("type") == "image"):
            out.append(u)
    return out


def layered_picture(eid: str, analysis: Analysis, m: TextMeasurer) -> str | None:
    """The picture the text was layered on in the source (ink on it at least
    LAYERED_MIN_ON; the hero first): the designer read it against that."""
    el = analysis.source_elements.get(eid)
    unit_key = analysis.element_unit.get(eid)
    if el is None or unit_key is None or el["type"] != "text":
        return None
    text_unit = analysis.unit(unit_key)
    try:
        ink = [visible_box(el, text_unit.members[eid], None, m)]
    except Exception:
        return None
    best: tuple[float, float, str] | None = None
    for u in _picture_units(analysis):
        on = _on_fraction(ink, [u.box])
        if on >= LAYERED_MIN_ON:
            rank = (1.0 if u.role == HERO else 0.0, on, u.key)
            if best is None or rank > best:
                best = rank
    return best[2] if best else None


def covering_unit(eid: str, analysis: Analysis, design: dict[str, Any], m: TextMeasurer) -> str | None:
    """The picture, plate or panel drawn under the text in `design` that
    covers at least BACKING_MIN_ON of its ink although the text did not sit
    on it in the source (the background excepted: it is everywhere)."""
    by_id = {e["id"]: e for e in design["elements"]}
    el = by_id.get(eid)
    src_el = analysis.source_elements.get(eid)
    unit_key = analysis.element_unit.get(eid)
    if el is None or src_el is None or unit_key is None or el["type"] != "text":
        return None
    text_unit = analysis.unit(unit_key)
    ink = [visible_box(el, Box.of(el), float(el["font_size"]), m)]
    try:
        ink_src = [visible_box(src_el, text_unit.members[eid], None, m)]
    except Exception:
        return None
    src = analysis.source_elements
    best: tuple[float, str] | None = None
    for u in analysis.units:
        if u.key == unit_key or u.role == BACKGROUND or not u.ids or not all(i in by_id for i in u.ids):
            continue
        if any(src.get(i, {}).get("type") == "text" for i in u.ids):
            continue
        frame = _frame(_members(u, by_id))
        on = _on_fraction(ink, [frame])
        if on >= BACKING_MIN_ON and _on_fraction(ink_src, [u.box]) < LAYERED_MIN_ON:
            if best is None or on > best[0]:
                best = (on, u.key)
    return best[1] if best else None


def lost_anchors(analysis: Analysis, design: dict[str, Any], target: Box) -> list[str]:
    """Content units that hugged an edge in the source and no longer do in
    `design` (the scorer's faith.anchors rule), most important first."""
    from engine.analyze import _anchors
    by_id = {el["id"]: el for el in design["elements"]}
    out: list[tuple[float, str]] = []
    for u in analysis.content_units():
        if u.anchor_x == "fluid" and u.anchor_y == "fluid":
            continue
        # Edge furniture only: a logo, a badge, the legal line, a button. The
        # headline, hero and copy are placed by the plan's composition; pinning
        # them to an edge is not a repair.
        if u.role not in PIN_ROLES:
            continue
        if not all(i in by_id for i in u.ids):
            continue
        frame = Box.union(rotated_extent(Box.of(by_id[i]), float(by_id[i].get("rotation", 0.0) or 0.0)) for i in u.ids)
        ax, ay = _anchors(frame, target)
        if (u.anchor_x != "fluid" and ax != u.anchor_x) or (u.anchor_y != "fluid" and ay != u.anchor_y):
            out.append((-u.importance, u.key))
    return [k for _, k in sorted(out)]


# ---------------------------------------------------------------------------
# directives: applied by engine.resize.realize to the materialized elements,
# before the rectifier (which then sorts out whatever the move disturbed)


def apply_directives(elements: list[dict[str, Any]], analysis: Analysis, target: Box,
                     directives: dict[str, Any], measurer: TextMeasurer | None = None) -> dict[str, Any]:
    """Apply `directives` in place; returns what was actually done (a note
    for the score). Directives:

      floor:   [text id, ...]      raise the text to its role's readable floor, re-wrapped on
                                   at most HEAD_MAX_LINES lines (one for a CTA) within the
                                   content width; a button label reshapes its pill around it
      widen:   [text id, ...]      widen the box until the text wraps on the allowed lines
      backing: {text id: unit key} the plate / panel covers the text with the source's padding
      onto:    {text id: hero key} the text unit moves onto the hero where it sat in the source
      quiet:   {text id: pic key}  (v10) the text unit moves to the band of the picture where its
                                   colour reads best (engine/backdrop.quiet_position)
      pin:     [unit key, ...]     the unit moves back to the canvas edge(s) it hugged in the source
      shrink:  {unit key: factor}  the unit scales about its centre (fonts follow)
      refit:   [text id, ...]      the font shrinks until the text fits its box
      cap_floor: True              read by resize._keep_headline_largest: the headline cap never
                                   pushes a text below its own floor
    """
    m = measurer or default_measurer()
    by_id = {el["id"]: el for el in elements}
    done: dict[str, Any] = {}
    for eid in directives.get("floor", []) or []:
        if eid in by_id and _raise_to_floor(eid, by_id, analysis, target, m):
            done.setdefault("floor", []).append(eid)
            if by_id[eid].get("rotation", 0.0) != (analysis.source_elements.get(eid) or {}).get("rotation", 0.0):
                done.setdefault("level", []).append(eid)
            key = analysis.element_unit.get(eid)
            if key is not None:
                # the raised unit is not the one that gives way when the rectifier must shrink something
                done.setdefault("protect", []).append(key)
    for eid in directives.get("widen", []) or []:
        if eid in by_id and _widen(eid, by_id, analysis, target, m):
            done.setdefault("widen", []).append(eid)
    for eid, key in (directives.get("backing") or {}).items():
        if eid in by_id and _backing(eid, key, by_id, analysis, target, m):
            done.setdefault("backing", {})[eid] = key
    for eid, key in (directives.get("onto") or {}).items():
        if eid in by_id and _onto(eid, key, by_id, analysis, target):
            done.setdefault("onto", {})[eid] = key
    for eid, key in (directives.get("quiet") or {}).items():
        if eid in by_id and _quiet(eid, key, by_id, analysis, target, m):
            done.setdefault("quiet", {})[eid] = key
            unit_key = analysis.element_unit.get(eid)
            if unit_key is not None:
                done.setdefault("protect", []).append(unit_key)
    for eid, key in (directives.get("off") or {}).items():
        if eid in by_id and _off(eid, key, by_id, analysis, target, m):
            done.setdefault("off", {})[eid] = key
    for key in directives.get("pin", []) or []:
        if _pin(key, by_id, analysis, target):
            done.setdefault("pin", []).append(key)
    for key, f in (directives.get("shrink") or {}).items():
        if _shrink(key, float(f), by_id, analysis):
            done.setdefault("shrink", {})[key] = f
    for eid in directives.get("refit", []) or []:
        if eid in by_id and _refit(eid, by_id, m):
            done.setdefault("refit", []).append(eid)
    if directives.get("cap_floor"):
        done["cap_floor"] = True
    return done


def cap_floors(analysis: Analysis, target: Box) -> dict[str, float]:
    """Per-role readable floors on `target` (for the cap_floor directive)."""
    met = Metrics.for_canvas(analysis.canvas, target, ratio_family(target))
    return {r: met.min_font(r) for r in (HEADLINE, CTA, SUBHEAD, BADGE, FINE_PRINT, "body", "text")}


def _margin(target: Box) -> float:
    return clamp(CONTENT_MARGIN_FRAC * min(target.w, target.h), 2.0, 24.0)


def _members(unit: Unit, by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_id[i] for i in unit.ids if i in by_id]


def _frame(els: Iterable[dict[str, Any]]) -> Box:
    return Box.union(rotated_extent(Box.of(e), float(e.get("rotation", 0.0) or 0.0)) for e in els)


def _move(els: Iterable[dict[str, Any]], dx: float, dy: float) -> None:
    for e in els:
        e["x"] = float(e["x"]) + dx
        e["y"] = float(e["y"]) + dy


def _scale(els: Iterable[dict[str, Any]], f: float, cx: float, cy: float) -> None:
    for e in els:
        b = Box.of(e)
        e["x"], e["y"] = cx + (b.x - cx) * f, cy + (b.y - cy) * f
        e["width"], e["height"] = b.w * f, b.h * f
        if e["type"] == "text":
            apply_font_size(e, max(1.0, float(e["font_size"]) * f))


def _place_text(el: dict[str, Any], w: float, h: float, target: Box) -> tuple[float, float]:
    """New top-left for a text box of size w x h that keeps the element's
    aligned edge (left / centre / right) where it was and stays on canvas.
    Returns the translation applied."""
    box = Box.of(el)
    align = str(el.get("text_align", "left") or "left")
    if align == "center":
        nx = box.cx - w / 2.0
    elif align == "right":
        nx = box.right - w
    else:
        nx = box.x
    ny = box.cy - h / 2.0
    mg = _margin(target)
    nx = (target.w - w) / 2.0 if w > target.w - 2 * mg else clamp(nx, mg, target.w - mg - w)
    ny = clamp(ny, 0.0, max(0.0, target.h - h))
    dx, dy = nx - box.x, ny - box.y
    el["x"], el["y"], el["width"], el["height"] = nx, ny, w, h
    return dx, dy


def _lines_allowed(role: str | None) -> int:
    return 1 if role == CTA else HEAD_MAX_LINES


def _raise_to_floor(eid: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box,
                    m: TextMeasurer) -> bool:
    el = by_id[eid]
    if el["type"] != "text":
        return False
    role = analysis.roles.get(eid)
    met = Metrics.for_canvas(analysis.canvas, target, ratio_family(target))
    floor = float(met.min_font(role or "text"))
    cur = float(el["font_size"])
    unit = _unit(analysis, eid)
    # A bar attached to the text that is taller / wider than the canvas (it
    # ran along a whole list in the source) would make the rectifier shrink
    # the whole unit to fit it, text included: trim it first.
    trimmed = _trim_attachments(unit, eid, by_id, target) if unit is not None else False
    if _tilted(el):
        # A vertical text (a spine label) whose line cannot run along the
        # canvas at the readable floor lies down: a designer sets the spine
        # text of a poster horizontal on a leaderboard (the rectifier would
        # otherwise shrink it to fit the strip's height). A diagonal text
        # keeps its angle and wraps onto shorter lines until its rotated
        # extent fits the canvas at the floor; spines that still fit keep
        # their rotation.
        if not _vertical(el):
            if _rewrap_tilted(el, floor, cur, _lines_allowed(role), target, m):
                return True
            if cur >= floor - 0.5:
                return trimmed
            # no wrap of the diagonal fits the canvas at the floor: it lies down too
        else:
            along = target.h - 2 * _margin(target)
            plain = dict(el, letter_spacing=0.0)
            if m.layout(plain, max(cur, floor), 1e7).max_width <= along + 0.5:
                return trimmed
        el["rotation"] = 0.0
        if unit is not None:
            for aid in unit.attachments:
                a = by_id.get(aid)
                if a is not None and _tilted(a):
                    a["rotation"] = 0.0
        if not _raise_to_floor(eid, by_id, analysis, target, m):
            # already at the floor: lay the box out horizontally at its size
            lay = m.layout(el, cur, max(1.0, target.w - 2 * _margin(target)))
            _place_text(el, max(1.0, lay.max_width + 1.0), math.ceil(lay.height) + 1.0, target)
        return True
    if cur >= floor - 0.5:
        return trimmed
    if unit is not None and _is_button(unit, analysis):
        return _reshape_button(unit, eid, floor, by_id, analysis, target, m) or trimmed
    full = target.w - 2 * _margin(target)
    font, ls, w, h = _fit_floor(el, floor, cur, _lines_allowed(role), full, m)
    if font is None:
        return trimmed
    old_c = (Box.of(el).cx, Box.of(el).cy)
    dx, dy = _place_text(el, w, h, target)
    apply_font_size(el, font)
    if ls is not None:
        el["letter_spacing"] = ls
    if unit is not None:
        # attached bars follow: scaled with the font about the text's old centre, then translated
        att = [by_id[a] for a in unit.attachments if a in by_id]
        _scale(att, font / max(cur, 1.0), old_c[0], old_c[1])
        _move(att, dx, dy)
        _trim_attachments(unit, eid, by_id, target)
    return True


def _trim_attachments(unit: Unit, eid: str, by_id: dict[str, dict[str, Any]], target: Box) -> bool:
    """Clamp each attachment of a host unit to the canvas (minus the margin),
    anchored on the text's side of it. Returns True when something changed."""
    changed = False
    mg = _margin(target)
    max_w, max_h = target.w - 2 * mg, target.h - 2 * mg
    text = by_id.get(eid)
    if text is None:
        return False
    tb = Box.of(text)
    for aid in unit.attachments:
        a = by_id.get(aid)
        if a is None:
            continue
        b = Box.of(a)
        if b.w <= max_w + 0.5 and b.h <= max_h + 0.5:
            continue
        w, h = min(b.w, max_w), min(b.h, max_h)
        # keep the edge nearest the text where it is
        x = b.x if abs(b.x - tb.x) <= abs(b.right - tb.right) else b.right - w
        y = b.y if abs(b.y - tb.y) <= abs(b.bottom - tb.bottom) else b.bottom - h
        a["x"], a["y"], a["width"], a["height"] = clamp(x, mg, max(mg, target.w - mg - w)), clamp(y, mg, max(mg, target.h - mg - h)), w, h
        changed = True
    return changed


def _fit_floor(el: dict[str, Any], floor: float, cur: float, lines: int, full: float, m: TextMeasurer
               ) -> tuple[float | None, float | None, float, float]:
    """The largest font in (cur, floor] at which the text wraps on at most
    `lines` lines within `full`, with the tracking it can afford: a letter-
    spaced headline keeps its tracking while the line fits and gives it up
    (down to none) before it gives up the readable size, the way a designer
    drops the tracking of a display word on a narrow strip. The renderer
    wraps on glyph widths, so the lines are the same whatever the spacing.
    Returns (font, letter_spacing or None when unchanged, box w, box h)."""
    plain = dict(el, letter_spacing=0.0)
    font = float(int(floor))
    while font > cur and m.layout(plain, font, full).max_width > full + 0.5:
        font -= 1.0
    if font <= cur + 0.5:
        return None, None, 0.0, 0.0
    w0 = _min_width_for_lines(m, plain, font, lines, full)
    lay0 = m.layout(plain, font, max(1.0, w0))
    ls_now = scaled_letter_spacing(el, font)
    ls: float | None = None
    if ls_now > 0.0:
        longest = max((len(line) for line in lay0.lines), default=1)
        afford = (full - lay0.max_width) / max(1, longest - 1)
        if ls_now > afford + 1e-6:
            ls = max(0.0, afford)
    probe = dict(el, letter_spacing=ls if ls is not None else ls_now, font_size=font)
    w = _min_width_for_lines(m, probe, font, lines, full)
    lay = m.layout(probe, font, max(1.0, w))
    w = max(w, lay.max_width + 1.0)
    return font, ls, w, math.ceil(lay.height) + 1.0


def _rewrap_tilted(el: dict[str, Any], floor: float, cur: float, lines: int, target: Box, m: TextMeasurer) -> bool:
    """A diagonal text at the floor font on the widest box whose rotated
    extent fits the canvas (minus the margin), on at most `lines` lines;
    the box stays centred where it was (the renderer rotates about the
    centre). False when no wrap fits or the text is already at the floor."""
    from engine.floorplan import _rotated_extent
    if cur >= floor - 0.5:
        return False
    rot = float(el.get("rotation", 0.0) or 0.0)
    mg = _margin(target)
    aw, ah = target.w - 2 * mg, target.h - 2 * mg
    font = float(int(floor))
    lo = max(1.0, m.layout(el, font, 1.0).max_width)          # the longest word
    hi = m.layout(el, font, 1e7).max_width + 1.0                # one line
    best = None
    for i in range(13):
        w = hi - (hi - lo) * i / 12.0
        lay = m.layout(el, font, w)
        if lay.n_lines > lines or lay.max_width > w + 0.5:
            continue
        bw, bh = lay.max_width + 1.0, math.ceil(lay.height) + 1.0
        ew, eh = _rotated_extent(bw, bh, rot)
        if ew <= aw + 0.5 and eh <= ah + 0.5:
            best = (bw, bh)
            break
    if best is None:
        return False
    box = Box.of(el)
    cx, cy = box.cx, box.cy
    bw, bh = best
    ew, eh = _rotated_extent(bw, bh, rot)
    cx = clamp(cx, mg + ew / 2.0, target.w - mg - ew / 2.0)
    cy = clamp(cy, mg + eh / 2.0, target.h - mg - eh / 2.0)
    el["x"], el["y"], el["width"], el["height"] = cx - bw / 2.0, cy - bh / 2.0, bw, bh
    apply_font_size(el, font)
    return True


def _is_button(unit: Unit, analysis: Analysis) -> bool:
    """A pill: shapes plus exactly one text that sits inside them (floorplan's
    button). A host unit (a text with bars attached beside it) is not one."""
    if unit.kind == "host" or len(unit.ids) < 2:
        return False
    src = analysis.source_elements
    texts = [i for i in unit.ids if src.get(i, {}).get("type") == "text"]
    shapes = [i for i in unit.ids if src.get(i, {}).get("type") == "shape"]
    if len(texts) != 1 or not shapes or len(texts) + len(shapes) != len(unit.ids):
        return False
    hull = Box.union(unit.members[s] for s in shapes if s in unit.members)
    lb = unit.members.get(texts[0])
    return lb is not None and hull.contains_fraction(lb) >= 0.8


def _reshape_button(unit: Unit, label: str, floor: float, by_id: dict[str, dict[str, Any]],
                    analysis: Analysis, target: Box, m: TextMeasurer) -> bool:
    """The label at the floor font on at most two lines, the pill re-shaped
    around it with the source's padding scaled to the font (what
    floorplan._Scaled._squeeze does inside a plan, here on a finished layout)."""
    el = by_id[label]
    src_el = analysis.source_elements[label]
    src_font = max(1.0, float(src_el["font_size"]))
    ub, lb = unit.box, unit.members[label]
    ink_src = m.layout(src_el, src_font, max(lb.w, 1.0)).max_width
    align = str(src_el.get("text_align", "left") or "left")
    ink_x = lb.x + (lb.w - ink_src) / 2.0 if align == "center" else (lb.right - ink_src if align == "right" else lb.x)
    pad_l, pad_r = max(0.0, ink_x - ub.x) / src_font, max(0.0, ub.right - (ink_x + ink_src)) / src_font
    pad_t, pad_b = max(0.0, lb.y - ub.y) / src_font, max(0.0, ub.bottom - lb.bottom) / src_font
    full = target.w - 2 * _margin(target)
    cur = float(el["font_size"])
    font, ls, _, _ = _fit_floor(el, floor, cur, 2, full - float(int(floor)), m)
    if font is None:
        return False
    probe = dict(el, letter_spacing=ls if ls is not None else scaled_letter_spacing(el, font))
    lay = m.layout(probe, font, max(1.0, full - font))
    ink_w = lay.max_width
    text_h = math.ceil(lay.height) + 1.0
    # the source's padding, scaled to the font, trimmed to the room the canvas leaves (floorplan._squeeze)
    room_w = max(0.0, (full - ink_w) / 2.0)
    room_h = max(0.0, (target.h - 2 * _margin(target) - text_h) / 2.0)
    pl, pr = max(0.5 * font, min(pad_l * font, room_w)), max(0.5 * font, min(pad_r * font, room_w))
    pt, pb = min(pad_t * font, room_h), min(pad_b * font, room_h)
    w, h = ink_w + pl + pr, text_h + pt + pb
    cur_frame = _frame(_members(unit, by_id))
    pill = Box(cur_frame.cx - w / 2.0, cur_frame.cy - h / 2.0, w, h)
    mg = _margin(target)
    pill = Box(clamp(pill.x, mg, max(mg, target.w - mg - w)), clamp(pill.y, 0.0, max(0.0, target.h - h)), w, h)
    sx, sy = pill.w / max(ub.w, 1e-6), pill.h / max(ub.h, 1e-6)
    for eid, mb in unit.members.items():
        e = by_id.get(eid)
        if e is None:
            continue
        if eid == label:
            e["x"], e["y"], e["width"], e["height"] = pill.x + pl - 0.5, pill.y + pt, ink_w + 1.0, text_h
            apply_font_size(e, font)
            if ls is not None:
                e["letter_spacing"] = ls
        else:
            e["x"], e["y"] = pill.x + (mb.x - ub.x) * sx, pill.y + (mb.y - ub.y) * sy
            e["width"], e["height"] = mb.w * sx, mb.h * sy
    return True


def _widen(eid: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box, m: TextMeasurer) -> bool:
    el = by_id[eid]
    if el["type"] != "text" or _tilted(el):
        return False
    role = analysis.roles.get(eid)
    font = float(el["font_size"])
    full = target.w - 2 * _margin(target)
    allowed = _lines_allowed(role)
    if m.layout(el, font, float(el["width"])).n_lines <= allowed:
        return False
    w = _min_width_for_lines(m, el, font, allowed, full)
    if w <= float(el["width"]) + 0.5:
        return False
    lay = m.layout(el, font, w)
    w = max(w, lay.max_width + 1.0)
    h = math.ceil(lay.height) + 1.0
    unit = _unit(analysis, eid)
    dx, dy = _place_text(el, w, h, target)
    if unit is not None:
        _move([by_id[a] for a in unit.attachments if a in by_id], dx, dy)
    return True


def _backing(eid: str, key: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box,
             m: TextMeasurer) -> bool:
    el = by_id[eid]
    plate = _unit(analysis, key)
    if plate is None or el["type"] != "text" or not all(i in by_id for i in plate.ids):
        return False
    src_el = analysis.source_elements.get(eid)
    text_unit = _unit(analysis, eid)
    if src_el is None or text_unit is None:
        return False
    font = float(el["font_size"])
    src_font = max(1.0, float(src_el["font_size"]))
    ink_src = visible_box(src_el, text_unit.members[eid], None, m)
    pb = plate.box
    pads = [max(0.0, ink_src.x - pb.x), max(0.0, ink_src.y - pb.y),
            max(0.0, pb.right - ink_src.right), max(0.0, pb.bottom - ink_src.bottom)]
    pads = [max(PAD_MIN_EM * font, p / src_font * font) for p in pads]
    ink = visible_box(el, Box.of(el), font, m)
    # Another panel / plate drawn between this plate and the text and now
    # covering the text (a card that grew over the heading that sat above
    # it in the source) is what the text would read against: slide the text
    # off it when a move keeps it on canvas, else cut the occluder back to
    # the far side of the text; failing both, the plate cannot help.
    occluder = _occluder(eid, plate, by_id, analysis, ink)
    if occluder is not None:
        if _off(eid, occluder.key, by_id, analysis, target, m):
            ink = visible_box(el, Box.of(el), float(el["font_size"]), m)
        elif not _cut_back(occluder, by_id, ink, PAD_MIN_EM * font):
            return False
    want = Box(ink.x - pads[0], ink.y - pads[1], ink.w + pads[0] + pads[2], ink.h + pads[1] + pads[3])
    cur = _frame(_members(plate, by_id))
    if plate.is_panel:
        new = Box.union([cur, want])
        new = _clip(_stick_edges(new, _edges_touched(pb, analysis.canvas), target), target)
    else:
        new = _clip(want, target)
        if not _is_rect(plate, analysis):
            f = max(new.w / max(pb.w, 1e-6), new.h / max(pb.h, 1e-6))
            w, h = pb.w * f, pb.h * f
            new = Box(new.cx - w / 2.0, new.cy - h / 2.0, w, h)
    ps = clamp(math.sqrt(target.area / analysis.canvas.area), 0.25, 2.5)
    for mid, mb in stack_boxes(plate, new, ps).items():
        e = by_id[mid]
        e["x"], e["y"], e["width"], e["height"] = mb.x, mb.y, mb.w, mb.h
    return True


def _z(analysis: Analysis, eid: str) -> float:
    el = analysis.source_elements.get(eid, {})
    if el.get("z_index") is not None:
        return float(el["z_index"])
    return float(list(analysis.source_elements).index(eid)) if eid in analysis.source_elements else 0.0


def _occluder(eid: str, plate: Unit, by_id: dict[str, dict[str, Any]], analysis: Analysis, ink: Box) -> Unit | None:
    """A non-text unit drawn above `plate` and below the text that covers at
    least BACKING_MIN_ON of the text's ink in the current layout but did not
    in the source."""
    z_text, z_plate = _z(analysis, eid), max(_z(analysis, i) for i in plate.ids)
    src_el = analysis.source_elements.get(eid)
    text_unit = _unit(analysis, eid)
    if src_el is None or text_unit is None:
        return None
    try:
        m = default_measurer()
        ink_src = visible_box(src_el, text_unit.members[eid], None, m)
    except Exception:
        return None
    src = analysis.source_elements
    for u in analysis.units:
        if u is plate or u.role == BACKGROUND or not u.ids or not all(i in by_id for i in u.ids):
            continue
        if any(src.get(i, {}).get("type") == "text" for i in u.ids):
            continue
        zu = max(_z(analysis, i) for i in u.ids)
        if not (z_plate < zu < z_text):
            continue
        frame = _frame(_members(u, by_id))
        if frame.contains_fraction(ink) >= BACKING_MIN_ON and u.box.contains_fraction(ink_src) < LAYERED_MIN_ON:
            return u
    return None


def _cut_back(occluder: Unit, by_id: dict[str, dict[str, Any]], ink: Box, gap: float) -> bool:
    """Shrink a rectangular occluder to the side of `ink` that keeps most of
    it (the text must lie in the occluder's outer third for the cut to make
    sense: a heading above a card, a caption beside it)."""
    if len(occluder.ids) != 1:
        return False
    e = by_id[occluder.ids[0]]
    if e["type"] != "shape" or float(e.get("rotation", 0.0) or 0.0):
        return False
    b = Box.of(e)
    cuts = [
        ("top", Box(b.x, ink.bottom + gap, b.w, b.bottom - (ink.bottom + gap))),
        ("bottom", Box(b.x, b.y, b.w, ink.y - gap - b.y)),
        ("left", Box(ink.right + gap, b.y, b.right - (ink.right + gap), b.h)),
        ("right", Box(b.x, b.y, ink.x - gap - b.x, b.h)),
    ]
    best = max(cuts, key=lambda c: c[1].area)
    nb = best[1]
    if nb.w <= 1.0 or nb.h <= 1.0 or nb.area < 0.6 * b.area:
        return False
    e["x"], e["y"], e["width"], e["height"] = nb.x, nb.y, nb.w, nb.h
    return True


def _onto(eid: str, key: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box) -> bool:
    hero = _unit(analysis, key)
    text_unit = _unit(analysis, eid)
    if hero is None or text_unit is None or not all(i in by_id for i in hero.ids):
        return False
    hb = _frame(_members(hero, by_id))
    if hb.w <= 0 or hb.h <= 0:
        return False
    # The text must read against the picture, so its ink stays inside it: a
    # single text that is wider than the picture is re-wrapped / shrunk to fit
    # (never below its floor; what still spills, the judge sees).
    el = by_id[eid]
    if el["type"] == "text" and not _tilted(el):
        m = default_measurer()
        pad = PAD_MIN_EM * float(el["font_size"])
        aw, ah = max(1.0, hb.w - 2 * pad), max(1.0, hb.h - 2 * pad)
        ink = visible_box(el, Box.of(el), float(el["font_size"]), m)
        if ink.w > aw + 0.5 or ink.h > ah + 0.5:
            role = analysis.roles.get(eid)
            met = Metrics.for_canvas(analysis.canvas, target, ratio_family(target))
            floor = float(met.min_font(role or "text"))
            cur = float(el["font_size"])
            font = cur
            while font > floor:
                lay = m.layout(el, font, aw)
                if lay.max_width <= aw + 0.5 and lay.height <= ah + 0.5 and lay.n_lines <= _lines_allowed(role):
                    break
                font -= 1.0
            lay = m.layout(el, font, aw)
            w = min(aw, max(1.0, lay.max_width + 1.0))
            old_c = (Box.of(el).cx, Box.of(el).cy)
            dx, dy = _place_text(el, w, math.ceil(lay.height) + 1.0, target)
            apply_font_size(el, font)
            att = [by_id[a] for a in text_unit.attachments if a in by_id]
            _scale(att, font / max(cur, 1.0), old_c[0], old_c[1])
            _move(att, dx, dy)
    tb = _frame(_members(text_unit, by_id))
    # the text sits where it sat on the hero in the source (centre as a fraction of the hero box)
    fx = clamp((text_unit.box.cx - hero.box.x) / max(hero.box.w, 1e-6), 0.0, 1.0)
    fy = clamp((text_unit.box.cy - hero.box.y) / max(hero.box.h, 1e-6), 0.0, 1.0)
    cx, cy = hb.x + fx * hb.w, hb.y + fy * hb.h
    if tb.w < hb.w:
        cx = clamp(cx, hb.x + tb.w / 2.0, hb.right - tb.w / 2.0)
    if tb.h < hb.h:
        cy = clamp(cy, hb.y + tb.h / 2.0, hb.bottom - tb.h / 2.0)
    cx = clamp(cx, tb.w / 2.0, target.w - tb.w / 2.0)
    cy = clamp(cy, tb.h / 2.0, target.h - tb.h / 2.0)
    _move(_members(text_unit, by_id), cx - tb.cx, cy - tb.cy)
    return True


def _quiet(eid: str, key: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box,
           m: TextMeasurer) -> bool:
    """v10: move the text unit to where it reads best against the picture
    `key` (engine/backdrop.quiet_position); False when nowhere reads better."""
    got = backdrop.quiet_position(eid, key, by_id, analysis, target, m)
    if got is None:
        return False
    dx, dy, _before, _after = got
    text_unit = _unit(analysis, eid)
    if text_unit is None:
        return False
    _move(_members(text_unit, by_id), dx, dy)
    return True


def _off(eid: str, key: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box,
         m: TextMeasurer) -> bool:
    """Slide the text unit off the picture `key` by the shortest move that
    keeps it on canvas (a gap of PAD_MIN_EM x the font between them)."""
    pic = _unit(analysis, key)
    text_unit = _unit(analysis, eid)
    if pic is None or text_unit is None or not all(i in by_id for i in pic.ids):
        return False
    el = by_id[eid]
    pb = _frame(_members(pic, by_id))
    members = _members(text_unit, by_id)
    tb = _frame(members)
    ink = visible_box(el, Box.of(el), float(el["font_size"]), m)
    if not ink.overlaps(pb):
        return False
    g = PAD_MIN_EM * float(el["font_size"]) + 1.0
    mg = _margin(target)
    moves = [(0.0, pb.bottom + g - ink.y), (0.0, -(ink.bottom + g - pb.y)),
             (pb.right + g - ink.x, 0.0), (-(ink.right + g - pb.x), 0.0)]
    best = None
    for dx, dy in moves:
        nb = Box(tb.x + dx, tb.y + dy, tb.w, tb.h)
        if nb.x < mg - 0.5 or nb.y < mg - 0.5 or nb.right > target.w - mg + 0.5 or nb.bottom > target.h - mg + 0.5:
            continue
        cost = abs(dx) + abs(dy)
        if best is None or cost < best[0]:
            best = (cost, dx, dy)
    if best is None:
        return False
    _move(members, best[1], best[2])
    return True


def _pin(key: str, by_id: dict[str, dict[str, Any]], analysis: Analysis, target: Box) -> bool:
    unit = _unit(analysis, key)
    if unit is None or not all(i in by_id for i in unit.ids):
        return False
    fb = _frame(_members(unit, by_id))
    mg = _margin(target)
    dx = dy = 0.0
    if unit.anchor_x == "left":
        dx = mg - fb.x
    elif unit.anchor_x == "right":
        dx = target.w - mg - fb.right
    elif unit.anchor_x == "center":
        dx = target.w / 2.0 - fb.cx
    if unit.anchor_y == "top":
        dy = mg - fb.y
    elif unit.anchor_y == "bottom":
        dy = target.h - mg - fb.bottom
    elif unit.anchor_y == "center":
        dy = target.h / 2.0 - fb.cy
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return False
    _move(_members(unit, by_id), dx, dy)
    return True


def _shrink(key: str, f: float, by_id: dict[str, dict[str, Any]], analysis: Analysis) -> bool:
    unit = _unit(analysis, key)
    if unit is None or not (0.0 < f < 1.0) or not all(i in by_id for i in unit.ids):
        return False
    fb = _frame(_members(unit, by_id))
    _scale(_members(unit, by_id), f, fb.cx, fb.cy)
    return True


def _refit(eid: str, by_id: dict[str, dict[str, Any]], m: TextMeasurer) -> bool:
    el = by_id[eid]
    if el["type"] != "text":
        return False
    w, h = float(el["width"]), float(el["height"])
    font = float(el["font_size"])
    if m.fits(el, font, w, h):
        return False
    apply_font_size(el, m.fit_font_size(el, w, h, font, 1.0))
    return True


MAX_TILT = 20.0                  # a text tilted up to this (degrees) is re-wrapped like a horizontal one


def _tilted(el: dict[str, Any]) -> bool:
    """Rotated too far for a horizontal re-wrap (a vertical or diagonal text)."""
    rot = abs(float(el.get("rotation", 0.0) or 0.0)) % 180.0
    return min(rot, 180.0 - rot) > MAX_TILT


def _vertical(el: dict[str, Any]) -> bool:
    rot = abs(float(el.get("rotation", 0.0) or 0.0)) % 180.0
    return abs(rot - 90.0) <= MAX_TILT
