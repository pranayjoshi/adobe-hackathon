"""Merged engine: our pipeline with Gaurav's strengths built in, kept apart from `engine/`.

`resize(source, target_canvas) -> design`. Nothing in `engine/` is edited: `install()`
adds two hooks to the pipeline at run time, and the final choice is re-ranked here.

1. Floor plans join the candidate pool (merged.floorplan): role-weighted layouts per
   canvas shape, built from our blocks, then repaired, checked and ranked like any other.
2. Template candidates crop eligible backgrounds to cover the canvas (the engine's own
   `cover_backdrops`, already allowed by its invariants) instead of shrinking them.
3. Ranking charges for wasted canvas: a background that no longer covers the canvas, or
   content shrunk to a fraction of its source share, counts as defects. Otherwise the
   uniform "Original arrangement" wins whenever every recomposed layout has one flaw.
4. When no layout passes every guardrail, the best structurally valid one is returned
   instead of refusing; when none is structurally valid, the one failing the fewest checks.
5. Letter spacing shrinks with the font (MERGED_SCALE_SPACING=0 turns this off).
6. Content may grow when the canvas grows (merged.growth): the strip-to-square fix
   (MERGED_GROWTH=0 turns this off).
7. Floor plans get leeway (MERGED_FLOORPLAN_LEEWAY=0 turns this off): they are excused the
   "moved away from the original" flags (text no longer on the part of the photo it began
   on), because recomposing is their point, and they win ties against the pipeline's own
   layouts. Real problems (collisions, overflow, lost contrast, unreadable or overlapping
   text, wasted canvas) still count in full, and only a floor plan whose headline is at
   least 85% as big as the best other layout's gets the leeway: bolder, never smaller.
8. The final choice is made by what the judges preferred (MERGED_PREFS=0 turns this off).
   The blind study's 600 votes showed that text size, headline size and wasted canvas decide
   the vote, while our defect counts barely do; ranking by defect count first is what kept
   choosing small, safe layouts. A shortlist (the ranking's top picks, the best floor plan, and
   Gaurav's layout from his engine, merged.gaurav_bridge, with any distorted image restored to
   its proportions) is scored by merged.prefs, a model fitted on those votes. Hard rules still
   veto: no distorted image, no text pushed off the canvas.
"""
from __future__ import annotations

import os
from typing import Any

LETTERBOX_DEFECTS = 2      # a background that no longer reaches the canvas edges
SPARSE_DEFECTS = 1         # content shrunk to well under its source share of the canvas
SPARSE_SHARE = 0.6
# Displacement from the source composition: a defect for a layout that keeps the source
# arrangement, a choice for a floor plan that recomposes it.
RECOMPOSITION_FLAGS = {'unsupported_text', 'scene_displacement'}
RECOMPOSITION_TIER2 = {'grouping.photo'}


def leeway() -> bool:
    return os.environ.get('MERGED_FLOORPLAN_LEEWAY', '1') != '0'


def is_floorplan(r) -> bool:
    return 'floorplan_' in r['template']  # includes the solver's and refiner's variants of one


def install() -> None:
    """Add the floor plans, the background cover and the growth rules (idempotent)."""
    import engine.pipeline as pipeline
    from engine.canonical import cover_backdrops
    from merged.floorplan import plans
    from merged import growth
    if os.environ.get('MERGED_GROWTH', '1') != '0':
        growth.install()  # strip-to-square: let text and blocks grow with the canvas
    if getattr(pipeline, '_merged_installed', False):
        return
    base_propose, base_serialise = pipeline.propose, pipeline.serialise

    def propose(A, cw, ch, emphasis='balanced', **kwargs):
        return base_propose(A, cw, ch, emphasis, **kwargs) + plans(A, cw, ch, emphasis)

    def serialise(source, target, P):
        return cover_backdrops(source, base_serialise(source, target, P))

    pipeline.propose, pipeline.serialise = propose, serialise
    pipeline._merged_installed = True


def _backgrounds(source):
    W, H = source['canvas']['width'], source['canvas']['height']
    bottom = min((e.get('z_index', 0) for e in source['elements']), default=0)
    return [e['id'] for e in source['elements']
            if e.get('z_index', 0) <= bottom + 1 and e['width'] * e['height'] >= 0.85 * W * H]


def wasted_canvas(source: dict[str, Any], design: dict[str, Any]) -> int:
    """Defects for unused canvas: a letterboxed background, or content shrunk far below
    the share of the canvas it used in the source."""
    from engine.pipeline import _content_share
    W, H = design['canvas']['width'], design['canvas']['height']
    by = {e['id']: e for e in design['elements']}
    backgrounds = [by[i] for i in _backgrounds(source) if i in by]
    if backgrounds:
        def covered(e):
            return max(0.0, min(W, e['x'] + e['width']) - max(0.0, e['x'])) * max(0.0, min(H, e['y'] + e['height']) - max(0.0, e['y']))
        if max(covered(e) for e in backgrounds) < 0.98 * W * H:
            return LETTERBOX_DEFECTS
    return SPARSE_DEFECTS if _content_share(design) < SPARSE_SHARE * _content_share(source) else 0


def headline_size(source, design):
    """Font size of the source's largest text in this design."""
    texts = [e for e in source['elements'] if e['type'] == 'text' and (e.get('content') or '').strip()]
    if not texts:
        return 0.0
    head = max(texts, key=lambda e: e['font_size'])['id']
    return float(next(e['font_size'] for e in design['elements'] if e['id'] == head))


HEADLINE_KEEP = 0.85   # a floor plan earns leeway only if its headline is this big vs the best other layout


def rank_key(source, r, eligible=True):
    """The pipeline's own ranking, with wasted canvas counted among the hard defects and,
    with leeway on, eligible floor plans excused recomposition flags and preferred on ties."""
    from engine.learning import preference_score
    from engine.refine import blocking
    rub = r.get('rubric')
    t2 = (rub or {}).get('tier2') or {}
    hard = r['quality']['defect_count'] + len(t2.get('defects', [])) + wasted_canvas(source, r['design'])
    plan = leeway() and is_floorplan(r) and eligible
    if plan:
        hard -= sum(len(v) for k, v in r['quality']['defects'].items() if k in RECOMPOSITION_FLAGS)
        hard -= sum(d['id'] in RECOMPOSITION_TIER2 for d in t2.get('defects', []))
    pref = preference_score(r)
    overall = t2.get('overall', r['score']['total']) if rub else r['score']['total']
    return (bool(r['valid'] and (rub['gate'] if rub else True)), -(blocking(r) if rub else 0), -hard,
            plan, bool(r.get('preferred_learned')), pref if pref is not None else overall)


def scale_spacing(source, design):
    """Letter spacing follows the font size down (never up: boxes were measured with it)."""
    src = {e['id']: e for e in source['elements']}
    for e in design['elements']:
        s = src.get(e['id'])
        if e['type'] == 'text' and s and s.get('letter_spacing') and s.get('font_size'):
            scaled = s['letter_spacing'] * e['font_size'] / s['font_size']
            if abs(scaled) < abs(e.get('letter_spacing') or 0):
                e['letter_spacing'] = round(scaled, 3)
    return design


def prefs_on() -> bool:
    return os.environ.get('MERGED_PREFS', '1') != '0'


SHORTLIST = 4   # the ranking's top picks that the judges' model chooses among


def unstretch(source, design):
    """Fit every image whose box lost its source proportions back inside that box, centred
    (position and size only). Cover-cropped canvas backgrounds are left as they are."""
    import copy
    src = {e['id']: e for e in source['elements']}
    out = copy.deepcopy(design)
    for e in out['elements']:
        s = src.get(e['id'])
        if e['type'] != 'image' or e.get('fit') == 'cover' or not s or min(s['width'], s['height'], e['width'], e['height']) <= 0:
            continue
        a = s['width'] / s['height']
        if abs(e['width'] / e['height'] / a - 1) <= 0.01:
            continue
        w, h = min(e['width'], e['height'] * a), min(e['height'], e['width'] / a)
        e.update(x=e['x'] + (e['width'] - w) / 2, y=e['y'] + (e['height'] - h) / 2, width=w, height=h)
    return out


def vetoed(r) -> bool:
    """Hard rules the judges' model may not trade away: a distorted image, text off the canvas."""
    for c in r['checks']:
        if not c['passed'] and c['name'] == 'Image proportions kept':
            return True
    d = r['design']; W, H = d['canvas']['width'], d['canvas']['height']
    for e in d['elements']:
        if e['type'] == 'text' and (e['x'] < -0.5 or e['y'] < -0.5 or e['x'] + e['width'] > W + 0.5 or e['y'] + e['height'] > H + 0.5):
            return True
    return False


def choose(source, target, results, top, key):
    """The shortlist's layout the judges' model scores highest (`top` when nothing beats it)."""
    from engine.canonical import cover_backdrops
    from engine.pipeline import analyze, assess_design
    from merged import gaurav_bridge, prefs
    shortlist = [top] + [r for r in results if r is not top][:SHORTLIST - 1]
    plans = [r for r in results if is_floorplan(r)]
    if plans and plans[0] not in shortlist:
        shortlist.append(plans[0])   # results are sorted, so this is the best floor plan
    theirs = gaurav_bridge.resize(source, target)
    if theirs:
        theirs = unstretch(source, cover_backdrops(source, {'canvas': dict(target), 'elements': theirs['elements']}))
        r = assess_design(source, target, theirs, analyze(source), "Gaurav's layout")
        r['template'] = 'gaurav'
        shortlist.append(r)
    allowed = [r for r in shortlist if not vetoed(r)] or [top]
    return max(allowed, key=lambda r: (prefs.score(source, r['design'], r), r is top))


def resize_with_report(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    import copy
    from engine.pipeline import generate
    install()
    target = {'width': int(target_canvas['width']), 'height': int(target_canvas['height'])}
    results = generate(source, target, all_candidates=True)
    if not results:
        raise ValueError('No candidate layout could be generated')
    # Leeway goes only to floor plans that keep the headline about as big as the best layout
    # the pipeline would have chosen without them: bolder, never smaller.
    others = sorted((r for r in results if not is_floorplan(r)), key=lambda r: rank_key(source, r), reverse=True)
    reference = headline_size(source, others[0]['design']) if others else 0.0
    def key(r):
        return rank_key(source, r, headline_size(source, r['design']) >= HEADLINE_KEEP * reference)
    results.sort(key=key, reverse=True)
    top = next((r for r in results if r['valid']), None)
    if top is None:
        # Every candidate fails some structural check (often one the source fails too, such as
        # a layer it already hides): keep the one failing the fewest, as the ranking orders them.
        top = min(results, key=lambda r: sum(not c['passed'] for c in r['checks']))
    if prefs_on():
        top = choose(source, target, results, top, key)
    design = copy.deepcopy(top['design'])
    if os.environ.get('MERGED_SCALE_SPACING', '1') != '0':
        design = scale_spacing(source, design)
    return {'design': design, 'template': top['template'], 'floorplan': is_floorplan(top), 'valid': bool(top['valid']),
            'passed_guardrails': bool(top['valid'] and (top.get('rubric') or {}).get('gate')),
            'wasted_canvas': wasted_canvas(source, top['design']), 'candidates': len(results)}


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    return resize_with_report(source, target_canvas)['design']
