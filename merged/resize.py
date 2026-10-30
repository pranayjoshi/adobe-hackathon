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
"""
from __future__ import annotations

import os
from typing import Any

LETTERBOX_DEFECTS = 2      # a background that no longer reaches the canvas edges
SPARSE_DEFECTS = 1         # content shrunk to well under its source share of the canvas
SPARSE_SHARE = 0.6


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


def rank_key(source, r):
    """The pipeline's own ranking, with wasted canvas counted among the hard defects."""
    from engine.learning import preference_score
    from engine.refine import blocking
    rub = r.get('rubric')
    t2 = (rub or {}).get('tier2') or {}
    hard = r['quality']['defect_count'] + len(t2.get('defects', [])) + wasted_canvas(source, r['design'])
    pref = preference_score(r)
    overall = t2.get('overall', r['score']['total']) if rub else r['score']['total']
    return (bool(r['valid'] and (rub['gate'] if rub else True)), -(blocking(r) if rub else 0), -hard,
            bool(r.get('preferred_learned')), pref if pref is not None else overall)


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


def resize_with_report(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    import copy
    from engine.pipeline import generate
    install()
    target = {'width': int(target_canvas['width']), 'height': int(target_canvas['height'])}
    results = generate(source, target, all_candidates=True)
    if not results:
        raise ValueError('No candidate layout could be generated')
    results.sort(key=lambda r: rank_key(source, r), reverse=True)
    top = next((r for r in results if r['valid']), None)
    if top is None:
        # Every candidate fails some structural check (often one the source fails too, such as
        # a layer it already hides): keep the one failing the fewest, as the ranking orders them.
        top = min(results, key=lambda r: sum(not c['passed'] for c in r['checks']))
    design = copy.deepcopy(top['design'])
    if os.environ.get('MERGED_SCALE_SPACING', '1') != '0':
        design = scale_spacing(source, design)
    return {'design': design, 'template': top['template'], 'valid': bool(top['valid']),
            'passed_guardrails': bool(top['valid'] and top['rubric']['gate']),
            'wasted_canvas': wasted_canvas(source, top['design']), 'candidates': len(results)}


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    return resize_with_report(source, target_canvas)['design']
