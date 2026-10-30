"""Candidate layouts for the judge loop (generate -> rectify -> render -> score).

One floor plan per ratio family (v1) becomes a handful of variants here:
hero left / right / stacked, badges in a row / in the corners / beside the
CTA, tight / normal / loose spacing, a slightly bolder or shyer global
scale, and (banners) the headline stacked above or inline with the subhead.
Small ratio changes get two or three anchor variants whose content scale
sits a little above min(sx, sy) when the canvas has room, plus a couple of
floor plans as alternatives. Every candidate is a complete `Placement`
(boxes for every element, font sizes via the measurer); `engine.resize`
materializes, rectifies, renders and scores them and keeps the best.

Nothing here looks at element ids; roles come from the analyzer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from engine.analyze import Analysis
from engine.floorplan import (
    BANNER,
    SQUARE,
    TALL,
    WIDE,
    PlanOptions,
    floorplan_relayout,
    is_strip,
    ratio_family,
)
from engine.geometry import Box
from engine.relayout import Placement, anchor_relayout, ratio_change, resolve_vertical_collisions
from engine.structure import StructureOptions, detect_structure, structure_relayout, structure_variants
from engine.textfit import TextMeasurer, default_measurer
from engine import critic, flags

logger = logging.getLogger(__name__)

SMALL_CHANGE_MAX = 1.75     # ratio_change() up to this: anchor variants compete unpenalised (score.ADAPT_RATIO)
MAX_CANDIDATES = 21         # v6: room for the 2-3 preserve-structure variants
ANCHOR_SCALE_BOOSTS = (1.0, 1.12, 1.25, 1.4)   # content scale = min(sx, sy) * boost, capped at max(sx, sy)
TIGHT, LOOSE = 0.75, 1.3
BOLD, BOLDER = 1.3, 1.5     # v4 scale_bias variants: ideal sizes past the proportional size when there is room
FILL = 1.5                  # the "fill" variant: one global scale, searched from 1.5 x proportional down
# S1: the perceived archetype (engine/perceive.py) reorders the plan variants
# and taxes the ones that fight it: editorial / quote designs prefer a text
# column over a hero-first plan, grids keep their chip rows. Without a
# perception (the default) nothing changes.
TEXT_FIRST = ("editorial", "quote")
ARCHETYPE_PENALTY = 3.0


@dataclass
class Candidate:
    name: str
    strategy: str                       # "anchor" | "floorplan" | "structure"
    placement: Placement
    options: PlanOptions | None = None  # floor plan variant
    scale: float | None = None          # anchor content scale
    fallback: bool = False              # safety net: pays the scorer's adaptation penalty, so it only
                                        # wins when every real candidate has a hard defect
    prior: float = 0.0                  # S1: soft penalty for a plan that fights the perceived archetype
    structure: StructureOptions | None = None   # v6 preserve-structure variant
    directives: dict[str, Any] | None = None    # v9: realize-time repairs (engine/critic.py)
    fix: str | None = None                      # v9: "<rule>: <recipe label>" for the notes


def generate_candidates(analysis: Analysis, source: dict[str, Any], target: Box,
                        measurer: TextMeasurer | None = None,
                        limit: int = MAX_CANDIDATES) -> list[Candidate]:
    m = measurer or default_measurer()
    by_id = {el["id"]: el for el in source["elements"]}
    change = ratio_change(analysis.canvas, target)
    family = ratio_family(target)
    archetype = (analysis.perception or {}).get("archetype") if getattr(analysis, "perception", None) else None
    variants = plan_variants(family, archetype)
    if flags.enabled("air") and family in (TALL, SQUARE, WIDE):
        # v11: whitespace allocated before the k search (PlanOptions.air)
        extra = [PlanOptions(air=True), PlanOptions(air=True, scale_bias=BOLD)]
        variants = variants[:3] + extra + variants[3:]
        limit += len(extra)
    if flags.enabled("strip") and is_strip(target):
        # v11 strip composer: triage first, the secondary copy and the badges
        # give way before the headline / logo / CTA (PlanOptions.rescue)
        extra = [PlanOptions(rescue=True), PlanOptions(rescue=True, scale_bias=BOLD),
                 PlanOptions(rescue=True, spacing=TIGHT)]
        variants = variants[:3] + extra + variants[3:]
        limit += len(extra)
    if flags.enabled("compose"):
        extra = compose_variants(family, target)
        variants = variants[:3] + extra + variants[3:]
        limit += len(extra)
    out: list[Candidate] = []

    small = change <= SMALL_CHANGE_MAX
    if small:
        for s, label in anchor_scales(analysis.canvas, target):
            out.append(Candidate(f"anchor {label} scale={s:.3f}", "anchor",
                                 _anchor(analysis, target, s), scale=s))
        plans = variants[:3] + [p for p in variants if p.scale_bias == BOLD][:1]
    else:
        plans = variants

    for opts in plans:
        pl = floorplan_relayout(analysis, target, by_id, m, opts)
        resolve_vertical_collisions(analysis, pl, target)
        out.append(Candidate(f"{family} {opts.name}", "floorplan", pl, options=opts,
                             prior=archetype_prior(family, opts, archetype)))

    # v6: preserve-structure variants (engine/structure.py): the source's
    # bands / split / lockup re-proportioned for the target instead of a
    # template; only for sources that have a zone to build a structure from.
    try:
        st = detect_structure(analysis, by_id)
    except Exception:
        logger.exception("structure detection failed; no structure candidates")
        st = None
    if st is not None:
        for sv in structure_variants(st, target, analysis.canvas):
            try:
                pl = structure_relayout(analysis, target, by_id, m, sv, st)
            except Exception:
                logger.exception("structure variant %s failed; skipped", sv.name)
                continue
            if pl is None:
                continue
            resolve_vertical_collisions(analysis, pl, target)
            out.append(Candidate(f"{family} structure {sv.name}", "structure", pl, structure=sv))

    if not small:
        # Safety candidate: proportions kept, everything slid toward its anchor.
        s = min(target.w / analysis.canvas.w, target.h / analysis.canvas.h)
        out.append(Candidate(f"anchor fallback scale={s:.3f}", "anchor",
                             _anchor(analysis, target, s), scale=s, fallback=True))

    return dedupe(out)[:limit]


MAX_REPAIRS = 6             # v9: new candidates per critic round


def generate_repairs(analysis: Analysis, source: dict[str, Any], target: Box,
                     diagnoses: list["critic.Diagnosis"], seeds: list[Candidate],
                     already: set[tuple], measurer: TextMeasurer | None = None,
                     limit: int = MAX_REPAIRS) -> list[Candidate]:
    """v9: candidates built from the critic's repair recipes (engine/critic.py).

    `seeds` are the judged candidates best first (no fallback). A recipe with
    plan overrides re-runs the floor plan of up to two floor-plan seeds with
    its PlanOptions changed (a structure recipe likewise for a structure
    seed); a recipe with directives alone reuses the seed's placement and
    hands the directives to `resize.realize`, which applies them before the
    rectifier. Repairs carry the seed's strategy and prior and are named
    "<seed> +fix:<recipe>"; `already` (placement key + directives) keeps the
    rounds from retrying a layout the judge has seen."""
    m = measurer or default_measurer()
    by_id = {el["id"]: el for el in source["elements"]}
    plan_seeds = [c for c in seeds if c.strategy == "floorplan" and c.options is not None][:2]
    struct_seeds = [c for c in seeds if c.strategy == "structure" and c.structure is not None][:1]
    direct_seeds = seeds[:critic.SEEDS]
    structure = None
    out: list[Candidate] = []
    for d in diagnoses:
        for r in d.recipes:
            jobs: list[tuple[str, Candidate]] = []
            if r.plan:
                jobs += [("plan", s) for s in plan_seeds]
            if r.structure:
                jobs += [("structure", s) for s in struct_seeds]
            if r.directives and (not jobs or not (r.plan or r.structure)):
                jobs += [("same", s) for s in direct_seeds]
            for mode, s in jobs:
                if not d.applies_to(s.name):
                    continue          # a hard diagnosis repairs the seeds that show the defect
                opts, st_opts, pl = s.options, s.structure, s.placement
                try:
                    if mode == "plan":
                        opts = replace(s.options, **r.plan)
                        pl = floorplan_relayout(analysis, target, by_id, m, opts)
                        resolve_vertical_collisions(analysis, pl, target)
                    elif mode == "structure":
                        if structure is None:
                            structure = detect_structure(analysis, by_id)
                        if structure is None:
                            continue
                        st_opts = replace(s.structure, **r.structure)
                        pl = structure_relayout(analysis, target, by_id, m, st_opts, structure)
                        if pl is None:
                            continue
                        resolve_vertical_collisions(analysis, pl, target)
                except Exception:
                    logger.exception("repair %s on %s failed; skipped", r.label, s.name)
                    continue
                directives = dict(s.directives or {})
                directives.update(r.directives)
                key = (placement_key(pl), critic._freeze(directives))
                if key in already:
                    continue
                already.add(key)
                out.append(Candidate(f"{s.name} +fix:{r.label}", s.strategy, pl, options=opts, scale=s.scale,
                                     prior=s.prior, structure=st_opts, directives=directives or None,
                                     fix=f"{d.rule}: {r.label}"))
                if len(out) >= limit:
                    return out
    return out


def anchor_scales(src: Box, target: Box) -> list[tuple[float, str]]:
    sx, sy = target.w / src.w, target.h / src.h
    lo, hi = min(sx, sy), max(sx, sy)
    scales: list[tuple[float, str]] = []
    for boost in ANCHOR_SCALE_BOOSTS:
        s = min(hi, lo * boost)
        if all(abs(s - t) > 0.015 * lo for t, _ in scales):
            scales.append((s, "fit" if boost == 1.0 else f"x{boost:g}"))
    return scales


def plan_variants(family: str, archetype: str | None = None) -> list[PlanOptions]:
    """Variants in rough order of prior usefulness (the list is truncated
    for small ratio changes). Duplicates that a design cannot realize
    (no chips, no hero) collapse in `dedupe`. With a perceived `archetype`
    the variants that suit it come first (stable: the rest keep their order)."""
    variants = _plan_variants(family)
    if archetype:
        variants = sorted(variants, key=lambda o: archetype_prior(family, o, archetype))
    return variants


def hero_first(family: str, opts: PlanOptions) -> bool:
    """Does this variant put the hero beside the text (hero-first) rather
    than in one column with it?"""
    if family == WIDE:
        return opts.hero_side != "stack"
    if family == SQUARE:
        return opts.hero_side in ("left", "right")
    if family == BANNER:
        return opts.text_flow == "column"
    return False


def archetype_prior(family: str, opts: PlanOptions, archetype: str | None) -> float:
    """Soft penalty for a plan variant that fights the perceived archetype."""
    if not archetype:
        return 0.0
    if archetype in TEXT_FIRST and hero_first(family, opts):
        return ARCHETYPE_PENALTY
    if archetype == "grid" and opts.chips != "row":
        return ARCHETYPE_PENALTY
    return 0.0


def compose_variants(family: str, target: Box) -> list[PlanOptions]:
    """v11: the compositions mined from the designer references (floorplan
    PlanOptions.compose). Wide: hero bleeding off its side edge, headline
    across the top. Banner: logo / CTA bookends, a two-tier strip, a bleeding
    hero (and, on a strip tall enough for a column, the bleeding wide plan)."""
    P = PlanOptions
    if family == WIDE:
        return [P(compose="bleed"), P(compose="header"), P(compose="bleed", scale_bias=BOLD)]
    if family == BANNER:
        out = [P(compose="bookend"), P(compose="twotier"), P(compose="bleed"),
               P(compose="bookend", hero_side="right")]
        if target.h >= 160:
            out.append(P(compose="bleed", text_flow="column"))
        return out
    return []


def _plan_variants(family: str) -> list[PlanOptions]:
    P = PlanOptions
    # v4: every family also tries bolder ideals (scale_bias 1.3 / 1.5) and a
    # "fill" variant (one global scale from 1.5 x proportional down); banners
    # also try the hero-beside-column plan (text_flow="column") and keep the
    # v3 stacked chip column as a variant (chips="stack").
    if family == TALL:
        return [P(), P(chips="corner"), P(chips="beside_cta"), P(spacing=TIGHT), P(spacing=LOOSE),
                P(scale_bias=1.15), P(scale_bias=BOLD), P(scale_bias=BOLDER), P(scale_bias=FILL, fill=True),
                P(scale_bias=0.85), P(chips="corner", spacing=TIGHT), P(chips="corner", scale_bias=BOLD)]
    if family == SQUARE:
        return [P(), P(chips="corner"), P(hero_side="left"), P(hero_side="right"),
                P(chips="beside_cta"), P(spacing=TIGHT), P(spacing=LOOSE), P(scale_bias=1.15),
                P(scale_bias=BOLD), P(scale_bias=BOLDER), P(scale_bias=FILL, fill=True),
                P(hero_side="left", chips="corner"), P(hero_side="right", chips="corner"),
                P(hero_side="left", scale_bias=BOLD)]
    if family == WIDE:
        return [P(), P(hero_side="left"), P(hero_side="right"), P(chips="corner"),
                P(chips="beside_cta"), P(hero_side="stack"), P(spacing=TIGHT), P(spacing=LOOSE),
                P(scale_bias=1.15), P(scale_bias=BOLD), P(scale_bias=BOLDER), P(scale_bias=FILL, fill=True),
                P(hero_side="left", chips="corner"), P(hero_side="right", chips="corner"),
                P(hero_side="left", scale_bias=BOLD), P(hero_side="right", scale_bias=BOLD),
                P(hero_side="stack", scale_bias=BOLD)]
    return [P(), P(text_flow="column"), P(text_flow="column", hero_side="right"), P(hero_side="right"),
            P(text_flow="inline"), P(hero_side="right", text_flow="inline"),
            P(spacing=TIGHT), P(spacing=LOOSE), P(scale_bias=1.15), P(scale_bias=BOLD),
            P(scale_bias=BOLDER), P(scale_bias=FILL, fill=True),
            P(text_flow="column", scale_bias=BOLD), P(text_flow="column", scale_bias=FILL, fill=True),
            P(chips="stack"), P(text_flow="inline", spacing=TIGHT)]


def _anchor(analysis: Analysis, target: Box, scale: float) -> Placement:
    pl = anchor_relayout(analysis, target, content_scale=scale)
    resolve_vertical_collisions(analysis, pl, target)
    return pl


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen: set[tuple] = set()
    out = []
    for c in cands:
        key = placement_key(c.placement)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def placement_key(pl: Placement) -> tuple:
    items = []
    for eid in sorted(pl.boxes):
        b = pl.boxes[eid]
        f = pl.font_size.get(eid, pl.font_scale.get(eid, 1.0))
        items.append((eid, round(b.x), round(b.y), round(b.w), round(b.h), round(f, 1)))
    return tuple(items)
