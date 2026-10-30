"""
CHAMELEON engine entry point: resize(source, target_canvas) -> design.

Pipeline (see docs/CONTEXT.md §10 in the parent folder for the full plan):

  1. analyze(source)      -> roles, groups, anchors, hierarchy          (engine/analyze.py)
  2. generate candidates  -> several complete layouts: anchor variants  (engine/candidates.py)
                             for small shape changes (relayout.py), floor
                             plan variants for big ones (floorplan.py)
  3. materialize each     -> boxes + font sizes fitted with the real brand
                             fonts (shrink only when the wrapped text would
                             overflow its box)
  4. rectify each         -> push apart collisions, keep things on canvas,
                             snap, even out gaps, keep group insets     (engine/rectify.py)
  5. render + score each  -> hard defects, readability (incl. contrast on
                             the rendered pixels), layout quality,
                             faithfulness to the source                 (engine/render.py, score.py)
  6. return the best      -> same ids, same order, rounded geometry

Hard rules honoured here: every id returned exactly once, no content /
font_family changes, images scaled uniformly (aspect kept), no asset pixels
touched. A uniform-scale fallback guarantees an answer even if everything
above raises.

Debugging: `resize_with_candidates()` returns every judged candidate with
its score breakdown (the playground shows them); setting the environment
variable CHAMELEON_DUMP_DIR writes each candidate's JSON + PNG + scores
there; CHAMELEON_NO_RENDER=1 (or CHAMELEON_RENDER=0) judges on geometry
alone. Rendering is a bonus signal only: engine/render.py switches it off
under CPU emulation, on slow renders and on repeated failures, so resize()
never depends on the renderer (engine/main.py writes the JSON only after
resize() returns).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.analyze import BACKGROUND, CTA, DECORATIVE, HEADLINE, Analysis, analyze
from engine.candidates import SMALL_CHANGE_MAX, Candidate, generate_candidates, generate_repairs, placement_key
from engine import critic, flags, visibility
from engine.geometry import Box
from engine import render as judge_render
from engine import optimize as opt
from engine.rectify import rectify, rotated_extent
from engine.floorplan import Metrics, ratio_family
from engine.relayout import Placement, grow_panels, ratio_change
from engine import role_merge
from engine.score import Score, score_design, structure_prior, structure_survives
from engine.textfit import MIN_FONT_SIZE, apply_font_size, default_measurer

logger = logging.getLogger(__name__)

JUDGE_RENDER = os.environ.get("CHAMELEON_NO_RENDER", "") not in ("1", "true", "yes")
TOP_RENDER = 3                     # candidates re-judged with pixels after the no-render pass
OPTIMIZE_TOP_K = 3                 # v7: seeds refined by the energy optimiser (CHAMELEON_OPTIMIZE_K overrides)
SAME_SHAPE_MAX = 1.02              # ... except when the target has the source's shape (ratio_change up to this)
SCARCE_FONT = 1.25                 # v8: a refinement never shrinks a headline / CTA font within this x of its floor
DUMP_DIR = os.environ.get("CHAMELEON_DUMP_DIR") or None
LAST_JUDGED: list["Judged"] = []     # every candidate of the most recent resize(), best first
LAST_ANALYSIS: Analysis | None = None   # the analysis the most recent resize() laid out (roles after the vote)
LAST_CRITIC: list[dict[str, Any]] = []  # v9: the critic rounds of the most recent resize() (diagnoses, repairs tried)
_run_counter = 0


@dataclass
class Judged:
    name: str
    strategy: str
    design: dict[str, Any]
    score: Score
    elapsed: float = 0.0
    fallback: bool = False
    roles: dict[str, Any] | None = None   # S1: who decided the roles (archetype, VLM status, voter counts)

    def summary(self) -> dict[str, Any]:
        out = {"name": self.name, "strategy": self.strategy, "elapsed": round(self.elapsed, 3),
               "fallback": self.fallback, "pixels": judge_render.status(), **self.score.as_dict()}
        if self.roles:
            out["roles"] = self.roles
        return out


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    target = Box(0.0, 0.0, float(target_canvas["width"]), float(target_canvas["height"]))
    try:
        best, judged = judge(source, target)
        _dump(judged)
        return best
    except Exception:  # pragma: no cover - last-resort safety net
        logger.exception("resize failed; falling back to uniform scale")
        return _uniform_fallback(source, target)


def resize_with_candidates(source: dict[str, Any], target_canvas: dict[str, Any]) -> tuple[dict[str, Any], list[Judged]]:
    """Like resize(), but also returns every judged candidate, best first."""
    target = Box(0.0, 0.0, float(target_canvas["width"]), float(target_canvas["height"]))
    try:
        return judge(source, target)
    except Exception:  # pragma: no cover
        logger.exception("resize failed; falling back to uniform scale")
        return _uniform_fallback(source, target), []


def judge(source: dict[str, Any], target: Box, render: bool | None = None) -> tuple[dict[str, Any], list[Judged]]:
    """Generate -> materialize -> rectify -> render -> score; best first."""
    global LAST_JUDGED, LAST_ANALYSIS
    analysis = analyze(source)
    judge_render.begin_task()          # per-task pixel budget; see engine/render.py
    understand(source, analysis)       # S1: VLM + classifier votes on the roles (no-op by default)
    LAST_ANALYSIS = analysis
    roles_info = roles_summary(analysis)
    render = JUDGE_RENDER if render is None else render
    judged: list[Judged] = []
    priors: dict[str, float] = {}
    cands_by_name: dict[str, Candidate] = {}
    # Stage 1: every candidate is judged on geometry and the JSON contrast
    # estimate (no pixels). Stage 2: the top few are re-judged with a render
    # without text, which refines contrast / busyness under each text.
    for cand in generate_candidates(analysis, source, target):
        t0 = time.time()
        try:
            design = realize(source, analysis, cand.placement, target)
            sc = score_design(analysis, source, design, target, render=False, strategy=cand.strategy)
            _apply_prior(sc, cand)
            sp = strip_prior(analysis, design, target)
            if sp:
                _apply_prior(sc, prior=sp, note="strip_prior")
        except Exception:
            logger.exception("candidate %s failed; skipped", cand.name)
            continue
        judged.append(Judged(cand.name, cand.strategy, design, sc, time.time() - t0, cand.fallback, roles_info))
        priors[cand.name] = cand.prior + sp
        cands_by_name[cand.name] = cand
    if not judged:
        raise RuntimeError("no candidate could be realized")
    # v8: when the source's own composition survives the shape change (a
    # preserve-structure candidate with no hard fail whose text still reads,
    # score.structure_survives), template layouts pay a small prior
    # (score.STRUCTURE_PRIOR); it rides in `priors` so refined copies and the
    # pixel re-judging carry it too.
    change = ratio_change(analysis.canvas, target)
    survives = {j.name: j.strategy == "structure" and not j.fallback and structure_survives(j.score) for j in judged}
    clean_structure = any(survives.values())
    for j in judged:
        sp = structure_prior(j.strategy, change, clean_structure, survives[j.name])
        if sp:
            _apply_prior(j.score, prior=sp, note="structure_prior")
            priors[j.name] = priors.get(j.name, 0.0) + sp
    # Best first; sort is stable, so earlier (default) candidates win exact
    # ties. The anchor safety net on a big shape change carries the scorer's
    # adaptation penalty, so it only wins when every real layout has a defect.
    judged.sort(key=lambda j: -j.score.total)
    # v9: the critic loop (engine/critic.py). When every candidate shares a
    # failure, or the best clean one is weak on a term with a repair, the
    # shared failure is diagnosed and repaired candidates are judged with the
    # rest; runs before the optimiser so the refinements start from the
    # repaired seeds. CHAMELEON_CRITIC=0 skips it: byte-identical output.
    if critic.enabled():
        judged = repair(judged, cands_by_name, analysis, source, target, priors, roles_info, change, clean_structure)
    # v7: the top seeds are walked down the design-principle energy
    # (engine/optimize.py); each refined layout competes as a candidate of
    # its own, so the judge still has the last word. CHAMELEON_OPTIMIZE=0
    # skips this and leaves the output byte-identical to the v6 engine.
    enabled, top_k = optimize_settings()
    if enabled and top_k > 0 and change > SAME_SHAPE_MAX:
        # A canvas of the same shape (identity, a pure scale) already holds a
        # finished design: nothing to re-arrange, the proportional copy is it.
        judged = refine(judged, analysis, source, target, priors, roles_info, top_k)
    # v11: the joint minimal-move constraint pass (engine/solve.py) on the
    # best few layouts; each solved layout competes as "<seed> +solve".
    if flags.enabled("solve") and change > SAME_SHAPE_MAX:
        judged = solve_pass(judged, analysis, source, target, priors, roles_info)
    if render:
        for j in judged[:TOP_RENDER]:
            t0 = time.time()
            try:
                j.score = score_design(analysis, source, j.design, target, render=True, strategy=j.strategy)
                _apply_prior(j.score, prior=priors.get(j.name, 0.0), note="prior")
            except Exception:
                logger.exception("re-judging %s with pixels failed; keeping the estimate", j.name)
            j.elapsed += time.time() - t0
        judged.sort(key=lambda j: -j.score.total)
    LAST_JUDGED = judged
    return judged[0].design, judged


STRIP_HEAD_LINES = 3         # v11 strip composer: on a banner <= floorplan.STRIP_MAX_H tall the headline
STRIP_LINES_PENALTY = 3.0    # reads on at most this many lines; a layout that wraps it further pays this prior


def strip_prior(analysis: Analysis, design: dict[str, Any], target: Box) -> float:
    """v11 (STRIP): the triage rule the judge only weakly knows. A headline
    wrapped on six 10 px lines in a 90 px leaderboard scored within 0.1 of
    the same headline on three lines; on a short banner every layout that
    wraps the headline beyond STRIP_HEAD_LINES pays STRIP_LINES_PENALTY."""
    from engine.floorplan import STRIP_MAX_H, WIDE_MAX_RATIO
    if not flags.enabled("strip") or target.h > STRIP_MAX_H or target.w / max(target.h, 1.0) <= WIDE_MAX_RATIO:
        return 0.0
    heads = [u for u in analysis.units if u.role == HEADLINE]
    if not heads:
        return 0.0
    eid = heads[0].primary or heads[0].ids[0]
    el = next((e for e in design["elements"] if e["id"] == eid), None)
    if el is None or el["type"] != "text" or abs(float(el.get("rotation", 0.0) or 0.0)) >= 1.0:
        return 0.0
    from engine.textfit import default_measurer
    if str(el.get("content", "")).count("\n") + 1 > STRIP_HEAD_LINES:
        return 0.0          # its own hard line breaks: nothing a layout can do about them
    return STRIP_LINES_PENALTY if default_measurer().layout(el).n_lines > STRIP_HEAD_LINES else 0.0


SOLVE_TOP_K = 1              # v11: the joint constraint pass polishes the winner; on the top 3 its few
                             # tenths of a point lifted a worse seed over the best one (Level 3, 728x90)


def solve_pass(judged: list[Judged], analysis: Analysis, source: dict[str, Any], target: Box,
               priors: dict[str, float], roles_info: dict[str, Any] | None, top_k: int = SOLVE_TOP_K
               ) -> list[Judged]:
    """v11: re-position the top `top_k` non-fallback layouts with the joint
    constraint solver (alignment, even rhythm, order, canvas) and add each
    changed one as a candidate; dropped when the judge finds a hard fail the
    seed did not have. Sizes and fonts are untouched by construction."""
    from engine.solve import solve
    extra: list[Judged] = []
    for j in [x for x in judged if not x.fallback][:top_k]:
        t0 = time.time()
        try:
            elements = solve(analysis, j.design["elements"], target)
            if elements is None:
                continue
            grow_panels(analysis, elements, target)
            _finalize(elements, analysis, target)
            design = {"canvas": {"width": int(target.w), "height": int(target.h)}, "elements": elements}
            sc = score_design(analysis, source, design, target, render=False, strategy=j.strategy)
            _apply_prior(sc, prior=priors.get(j.name, 0.0), note="prior")
            seed_hard = {_hard_key(h) for h in j.score.hard}
            if any(_hard_key(h) not in seed_hard for h in sc.hard):
                continue
        except Exception:
            logger.exception("solving %s failed; skipped", j.name)
            continue
        name = f"{j.name} +solve"
        priors[name] = priors.get(j.name, 0.0)
        extra.append(Judged(name, j.strategy, design, sc, time.time() - t0 + j.elapsed, False, roles_info))
    if extra:
        judged = sorted(judged + extra, key=lambda x: -x.score.total)
    return judged


def optimize_settings() -> tuple[bool, int]:
    """(enabled, top-K) from the environment, read per call so tests can
    flip it: CHAMELEON_OPTIMIZE=0 disables, CHAMELEON_OPTIMIZE_K sets K."""
    flag = os.environ.get("CHAMELEON_OPTIMIZE", "1").strip().lower()
    enabled = flag not in ("0", "false", "no", "off")
    try:
        k = int(os.environ.get("CHAMELEON_OPTIMIZE_K", "") or OPTIMIZE_TOP_K)
    except ValueError:
        k = OPTIMIZE_TOP_K
    return enabled, k


def refine(judged: list[Judged], analysis: Analysis, source: dict[str, Any], target: Box,
           priors: dict[str, float], roles_info: dict[str, Any] | None, top_k: int = OPTIMIZE_TOP_K
           ) -> list[Judged]:
    """Optimise the top `top_k` non-fallback seeds and add each changed
    result as a candidate "<seed> +opt" (same strategy and prior), then
    re-sort. A refined layout is dropped when the judge finds a hard fail
    its seed did not have (the optimiser's own invariant covers its energy;
    this covers the final rounding and the judge's exact rules), or when it
    spent legibility that was already scarce (v8): a headline / CTA font
    within SCARCE_FONT x of its readable floor may not get smaller than in
    the seed (an aesthetic move is never worth a 7 px button on a banner)."""
    extra: list[Judged] = []
    met = Metrics.for_canvas(analysis.canvas, target, ratio_family(target))
    seeds = [x for x in judged if not x.fallback][:top_k]
    # v9: a critic repair in the top K must not push the best plain seed out
    # of the refinement (its refined copy was the winner in 1 of 8 cases
    # before); one extra seed at most.
    plain = next((x for x in judged if not x.fallback and "+fix:" not in x.name), None)
    if plain is not None and plain not in seeds:
        seeds.append(plain)
    for j in seeds:
        t0 = time.time()
        try:
            res = opt.optimize(analysis, source, j.design["elements"], target)
            if not res.changed:
                continue
            elements = res.elements
            grow_panels(analysis, elements, target)
            _keep_headline_largest(elements, source, analysis)
            _finalize(elements, analysis, target)
            design = {"canvas": {"width": int(target.w), "height": int(target.h)}, "elements": elements}
            sc = score_design(analysis, source, design, target, render=False, strategy=j.strategy)
            _apply_prior(sc, prior=priors.get(j.name, 0.0), note="prior")
            seed_hard = {_hard_key(h) for h in j.score.hard}
            if any(_hard_key(h) not in seed_hard for h in sc.hard):
                logger.debug("refined %s dropped: new hard fail %s", j.name, sc.hard)
                continue
            if _spent_scarce_font(j.design["elements"], elements, analysis, met):
                logger.debug("refined %s dropped: shrank a headline / CTA at its floor", j.name)
                continue
            sc.notes["optimize"] = res.summary()
            name = f"{j.name} +opt"
            extra.append(Judged(name, j.strategy, design, sc, time.time() - t0, j.fallback, roles_info))
            priors[name] = priors.get(j.name, 0.0)
        except Exception:
            logger.exception("optimising %s failed; seed kept", j.name)
    if extra:
        judged = judged + extra
        judged.sort(key=lambda j: -j.score.total)
    return judged


def repair(judged: list[Judged], cands_by_name: dict[str, Candidate], analysis: Analysis,
           source: dict[str, Any], target: Box, priors: dict[str, float], roles_info: dict[str, Any] | None,
           change: float, clean_structure: bool) -> list[Judged]:
    """v9 critic loop (engine/critic.py): up to `critic.settings()` rounds of
    diagnose -> regenerate -> judge. Each round reads the shared failures of
    the judged candidates, builds at most `candidates.MAX_REPAIRS` repair
    candidates from the recipes (seeded from the best real candidates, and
    from the previous round's repairs that removed what they were built for),
    and judges them with the rest, priors included. Stops when the best
    candidate is clean and reaches the target total, when nothing is
    diagnosed, or when a round adds nothing new. The rounds are recorded in
    LAST_CRITIC and in the winner's notes["critic_rounds"]; every repair's
    notes["critic"] says which round, recipe and directives made it."""
    global LAST_CRITIC
    rounds, goal = critic.settings()
    log: list[dict[str, Any]] = []
    LAST_CRITIC = log
    already: set[tuple] = {(placement_key(c.placement), critic._freeze(c.directives or {})) for c in cands_by_name.values()}
    progress: list[Candidate] = []
    for rnd in range(1, rounds + 1):
        best = judged[0]
        if not best.fallback and not best.score.hard and best.score.total >= goal:
            break
        diags = critic.diagnose(judged, analysis, source, target)
        if not diags:
            break
        seeds = progress + [cands_by_name[j.name] for j in judged
                            if not j.fallback and j.name in cands_by_name and cands_by_name[j.name] not in progress]
        new = generate_repairs(analysis, source, target, diags, seeds[:4], already)
        if not new:
            break
        diagnosed = {critic.hard_kind_of(d) for d in diags if d.hard}
        progress = []
        added: list[dict[str, Any]] = []
        for cand in new:
            t0 = time.time()
            try:
                design = realize(source, analysis, cand.placement, target, directives=cand.directives)
                sc = score_design(analysis, source, design, target, render=False, strategy=cand.strategy)
                _apply_prior(sc, cand)
                sp = structure_prior(cand.strategy, change, clean_structure,
                                     cand.strategy == "structure" and structure_survives(sc))
                if sp:
                    _apply_prior(sc, prior=sp, note="structure_prior")
                lp = strip_prior(analysis, design, target)
                if lp:
                    _apply_prior(sc, prior=lp, note="strip_prior")
                    sp += lp
            except Exception:
                logger.exception("repair %s failed; skipped", cand.name)
                continue
            sc.notes["critic"] = {"round": rnd, "fix": cand.fix, "directives": cand.directives or {}}
            judged.append(Judged(cand.name, cand.strategy, design, sc, time.time() - t0, False, roles_info))
            priors[cand.name] = cand.prior + sp
            cands_by_name[cand.name] = cand
            added.append({"name": cand.name, "total": round(sc.total, 2), "hard": len(sc.hard)})
            if diagnosed and not (diagnosed & {critic._hard_kind(h) for h in sc.hard}):
                progress.append(cand)
        before = judged[0].name
        judged.sort(key=lambda j: -j.score.total)
        log.append({"round": rnd, "diagnoses": [d.summary() for d in diags], "tried": added,
                    "best_before": before, "best_after": judged[0].name})
        if not added:
            break
    if log:
        judged[0].score.notes["critic_rounds"] = log
    return judged


def _spent_scarce_font(seed: list[dict[str, Any]], refined: list[dict[str, Any]], analysis: Analysis,
                       met: Metrics) -> bool:
    """Did the refinement make a headline / CTA that was already within
    SCARCE_FONT x of its readable floor smaller than it was in the seed?"""
    before = {el["id"]: float(el["font_size"]) for el in seed if el["type"] == "text"}
    for el in refined:
        if el["type"] != "text" or el["id"] not in before:
            continue
        role = analysis.roles.get(el["id"])
        if role not in (HEADLINE, CTA):
            continue
        was, now = before[el["id"]], float(el["font_size"])
        if now < was - 1e-6 and was < SCARCE_FONT * met.min_font(role):
            return True
    return False


def _hard_key(h: str) -> str:
    """A hard-fail message without its numbers (areas, ratios), so the same
    defect on the same units compares equal before and after a refinement."""
    return "".join(ch for ch in h if not (ch.isdigit() or ch in ".%"))


def understand(source: dict[str, Any], analysis: Analysis) -> None:
    """S1: let the vision model (CHAMELEON_VLM=1) and the learned classifier
    (CHAMELEON_ROLE_MODEL=1) vote on the roles the rules inferred. Both are
    off by default, and then nothing here touches the analysis: the output
    stays byte-identical to the rule-only engine. Never raises."""
    perception = None
    try:
        if os.environ.get("CHAMELEON_VLM", "").strip().lower() in ("1", "true", "yes", "on"):
            from engine import perceive            # imports engine.vlm only when switched on
            perception = perceive.perceive(source, analysis)
    except Exception:  # pragma: no cover - perceive() already swallows everything
        logger.exception("perception failed; roles from the rules")
        perception = None
    try:
        if perception is not None or role_merge.enabled():
            role_merge.apply_votes(analysis, source, perception)
    except Exception:  # pragma: no cover
        logger.exception("role vote failed; roles from the rules")


def roles_summary(analysis: Analysis) -> dict[str, Any] | None:
    """Compact record for Judged.summary(): archetype, VLM status, how many
    units each voter decided, how many roles changed."""
    p = analysis.perception
    if not p:
        return None
    return {"archetype": p.get("archetype"), "vlm": p.get("vlm"), "by": dict(p.get("by") or {}),
            "changed": p.get("changed", 0)}


def _apply_prior(sc: Score, cand: Candidate | None = None, prior: float | None = None,
                 note: str = "archetype_penalty") -> None:
    """A candidate's archetype prior (engine/candidates.py) or structure prior
    (engine/score.py, v8) is a soft penalty outside the 0-100 breakdown, like
    the adaptation penalty. `note` names it in the score's notes ("prior" when
    a re-scored candidate gets the sum of its priors back in one go)."""
    v = float(cand.prior if cand is not None else (prior or 0.0))
    if v:
        sc.total -= v
        sc.penalty += v
        sc.notes[note] = round(v, 2)


def realize(source: dict[str, Any], analysis: Analysis, placement: Placement, target: Box,
            directives: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn a Placement into a finished design: fit the text, fix defects,
    round and clamp. `directives` (v9, engine/critic.py) are applied to the
    materialized elements before the rectifier: the critic's repairs."""
    elements = materialize(source, analysis, placement, target)
    applied: dict[str, Any] = {}
    protect: set[str] = set()
    if directives:
        applied = critic.apply_directives(elements, analysis, target, directives)
        # the texts a floor recipe names are protected whether or not they needed raising here
        protect = set(applied.get("protect", ())) | {analysis.element_unit[e] for e in directives.get("floor") or ()
                                                     if e in analysis.element_unit}
    elements = rectify(analysis, elements, target, protect=protect)
    if directives and directives.get("floor"):
        # The rectifier may have scaled the text down again (a unit wider than
        # the strip, a collision nothing could move away from) or the plan
        # had it at the floor and the rectifier took it under: raise it once
        # more and let a short rectify pass settle what that disturbed.
        if critic.apply_directives(elements, analysis, target, {"floor": directives["floor"]}).get("floor"):
            elements = rectify(analysis, elements, target, passes=2, protect=protect)
    grow_panels(analysis, elements, target)   # zones follow the content the rectifier moved
    if flags.enabled("visible"):
        # v12: content the source showed is never drawn under a panel that grew
        # over it (engine/visibility.py); the judge fails whatever remains.
        visibility.reveal(analysis, elements, target)
    _keep_headline_largest(elements, source, analysis,
                           floors=critic.cap_floors(analysis, target) if applied.get("cap_floor") else None)
    _finalize(elements, analysis, target)
    return {"canvas": {"width": int(target.w), "height": int(target.h)}, "elements": elements}


def materialize(source: dict[str, Any], analysis: Analysis, placement: Placement, target: Box) -> list[dict[str, Any]]:
    elements = []
    for el in source["elements"]:
        new = copy.deepcopy(el)
        box = placement.boxes[el["id"]]
        new["x"], new["y"], new["width"], new["height"] = box.x, box.y, box.w, box.h
        if el["type"] == "text":
            if el["id"] in placement.leading:
                new["line_height"] = placement.leading[el["id"]]
            if el["id"] in placement.tracking and new.get("letter_spacing"):
                new["letter_spacing"] = float(new["letter_spacing"]) * placement.tracking[el["id"]]
            apply_font_size(new, _fit_text(new, el, placement.font_scale.get(el["id"], 1.0),
                                           placement.font_size.get(el["id"])))
        elements.append(new)
    return elements


def _fit_text(new: dict[str, Any], src: dict[str, Any], scale: float,
              explicit: float | None = None) -> float:
    """Scale the font with its box, then shrink only if the wrapped text would
    overflow. Grows the box height instead when the text needs a little more
    room than the box has (the renderer does not clip, so the box height is
    only a layout hint, but the judge sees overflow as a collision).

    `explicit` is a size a strategy already fitted with the measurer (floor
    plans); it is trusted below MIN_FONT_SIZE too, since tiny badge labels
    on a tiny banner are better than labels spilling out of their badge."""
    measurer = default_measurer()
    w, h = float(new["width"]), float(new["height"])
    if explicit is not None:
        wanted = max(1.0, float(explicit))
        if measurer.fits(new, wanted, w, h):
            return float(int(wanted)) if wanted >= 1.0 else wanted
        size = measurer.fit_font_size(new, w, h, wanted, min(wanted, 1.0))
        if size < 0.8 * wanted and flags.enabled("strip"):
            # v11: the plan measured this size; a box a little short (wrap
            # differences on tracked text) grows instead of halving the font
            needed = measurer.natural_height(new, wanted, w)
            if needed <= h * 1.6 and measurer.layout(new, wanted, w).max_width <= w + 0.5:
                new["height"] = needed
                return float(int(wanted))
        return size
    wanted = max(MIN_FONT_SIZE, float(src["font_size"]) * scale)
    if measurer.fits(new, wanted, w, h):
        return float(int(wanted))
    size = measurer.fit_font_size(new, w, h, wanted, MIN_FONT_SIZE)
    if size < 0.8 * wanted:
        # Prefer a slightly taller box over a much smaller font: text boxes
        # are usually drawn with generous height in the source anyway.
        needed = measurer.natural_height(new, wanted, w)
        if needed <= h * 1.35:
            new["height"] = needed
            return float(int(wanted))
    return size


def _keep_headline_largest(elements: list[dict[str, Any]], source: dict[str, Any], analysis: Analysis,
                           floors: dict[str, float] | None = None) -> None:
    """The headline stays the largest text in the output, as it was in the
    source: any text that was smaller than the headline there and comes out
    bigger here (a short label the fitter could grow, a badge number) is
    capped at the headline's size. Text the designer made bigger than the
    headline (a date in a sticker) keeps that right. With `floors` (v9: the
    critic's cap_floor directive) the cap never pushes a text below its own
    role's readable floor when the headline itself ended under it."""
    src = {el["id"]: el for el in source["elements"]}
    heads = [el for el in elements if analysis.roles.get(el["id"]) == HEADLINE and el["type"] == "text"]
    if not heads:
        return
    head = heads[0]
    h_src = float(src[head["id"]]["font_size"])
    h_out = float(head["font_size"])
    for el in elements:
        if el["type"] != "text" or el is head or el["id"] not in src:
            continue
        if float(src[el["id"]]["font_size"]) < h_src and float(el["font_size"]) > h_out:
            cap = h_out if floors is None else max(h_out, float(floors.get(analysis.roles.get(el["id"], ""), 0.0)))
            if float(el["font_size"]) > cap:
                apply_font_size(el, cap)


def _finalize(elements: list[dict[str, Any]], analysis: Analysis, target: Box) -> None:
    for el in elements:
        role = analysis.roles.get(el["id"])
        rot = float(el.get("rotation", 0.0) or 0.0)
        # Keep important things fully on canvas; background/decoration may bleed.
        if role not in (BACKGROUND, DECORATIVE) and not rot:
            el["width"] = min(el["width"], target.w)
            el["height"] = min(el["height"], target.h)
            el["x"] = min(max(el["x"], 0.0), target.w - el["width"])
            el["y"] = min(max(el["y"], 0.0), target.h - el["height"])
        elif role not in (BACKGROUND, DECORATIVE):
            # Rotated: the box is in the element's own frame; what must stay
            # on canvas is its rotated extent (the renderer rotates about the
            # box centre), so slide it rather than clipping the frame box.
            ext = rotated_extent(Box.of(el), rot)
            if el["type"] != "text" and (ext.w > target.w + 0.01 or ext.h > target.h + 0.01):
                # A shape or image whose rotated extent is bigger than the
                # canvas (a 1-degree panel stretched to the full height, v5)
                # shrinks about its centre until the extent fits.
                f = min(target.w / max(ext.w, 1e-6), target.h / max(ext.h, 1e-6))
                cx, cy = float(el["x"]) + float(el["width"]) / 2.0, float(el["y"]) + float(el["height"]) / 2.0
                el["width"], el["height"] = float(el["width"]) * f, float(el["height"]) * f
                el["x"], el["y"] = cx - el["width"] / 2.0, cy - el["height"] / 2.0
                ext = rotated_extent(Box.of(el), rot)
            nx = min(max(ext.x, 0.0), max(0.0, target.w - ext.w))
            ny = min(max(ext.y, 0.0), max(0.0, target.h - ext.h))
            el["x"] = float(el["x"]) + (nx - ext.x)
            el["y"] = float(el["y"]) + (ny - ext.y)
        for k in ("x", "y"):
            el[k] = round(float(el[k]), 2)
        for k in ("width", "height"):
            # Text boxes round *up*: a width 0.005 px smaller re-wraps a line
            # that was fitted exactly, and then the text overflows.
            v = float(el[k])
            el[k] = math.ceil(v * 100.0 - 1e-6) / 100.0 if el["type"] == "text" else round(v, 2)
        if el["type"] == "text":
            el["font_size"] = max(1.0, round(float(el["font_size"]), 2))
            if el.get("letter_spacing"):
                el["letter_spacing"] = round(float(el["letter_spacing"]), 3)
            # cr-renderer validates TextElement width/height > 0 and rejects
            # the whole design otherwise (no PNG at all). A slot squeezed to
            # nothing on a 728x90 banner must still be a 1px box.
            el["width"] = max(el["width"], 1.0)
            el["height"] = max(el["height"], 1.0)
            if not rot:
                el["x"] = min(el["x"], max(0.0, target.w - el["width"]))
                el["y"] = min(el["y"], max(0.0, target.h - el["height"]))


def _uniform_fallback(source: dict[str, Any], target: Box) -> dict[str, Any]:
    sx = target.w / source["canvas"]["width"]
    sy = target.h / source["canvas"]["height"]
    s = min(sx, sy)
    out = []
    for el in source["elements"]:
        el = copy.deepcopy(el)
        el["x"], el["y"] = el["x"] * sx, el["y"] * sy
        el["width"], el["height"] = el["width"] * s, el["height"] * s
        if el["type"] == "text":
            apply_font_size(el, max(1.0, el["font_size"] * s))
        out.append(el)
    return {"canvas": {"width": int(target.w), "height": int(target.h)}, "elements": out}


def _dump(judged: list[Judged]) -> None:
    """CHAMELEON_DUMP_DIR: write every candidate (JSON, PNG, scores) of each
    resize() call into its own numbered folder. Dev aid, off by default."""
    global _run_counter
    if not DUMP_DIR:
        return
    _run_counter += 1
    folder = Path(DUMP_DIR) / f"run{_run_counter:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        from engine.render import render_design
    except Exception:  # pragma: no cover
        render_design = None  # type: ignore
    rows = []
    for i, j in enumerate(judged):
        stem = folder / f"cand{i:02d}"
        stem.with_suffix(".json").write_text(json.dumps(j.design, indent=2))
        if render_design is not None:
            img = render_design(j.design, max_size=1200)
            if img is not None:
                img.save(stem.with_suffix(".png"))
        rows.append({"file": stem.name, **j.summary()})
    (folder / "scores.json").write_text(json.dumps(rows, indent=2))
