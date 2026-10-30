"""`resize(source, target_canvas) -> design`: the submission engine.

1. Generate broadly: our pipeline with the merged engine's floor plans and growth rules, plus
   Gaurav's best layouts, each rebuilt by final.comply so it obeys the structural rules.
2. Gate: only layouts failing the fewest structural checks stay, then those losing the fewest
   texts to tiny type; then new text overflow, text printed over text, newly hidden elements and
   lost contrast are ruled out, when anything is left.
3. Judge: the survivors are drawn by the official renderer and the model fitted on the blind
   panel's votes picks one. It never refuses: when everything fails, the least bad layout returns.

Unlike merged/, letter spacing is never changed: the FAQ allows position, size and font size only.
"""
from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from final import gaurav, pixels, prefs
from final.comply import comply

SHORTLIST = 5        # the pipeline's top layouts by its own ranking
REFINE_SECONDS = float(os.environ.get('FINAL_REFINE_SECONDS', 12))   # the pipeline's hill-climb budget
TEXT_OVERPRINT = 0.15   # share of the smaller text box newly covered by another text
MIN_TEXT_PX = 3         # type below this is gone to the eye, though its element is still there


def _pairs(design):
    texts = [e for e in design['elements'] if e['type'] == 'text' and str(e.get('content') or '').strip()]
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            w = min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x'])
            h = min(a['y'] + a['height'], b['y'] + b['height']) - max(a['y'], b['y'])
            area = min(a['width'] * a['height'], b['width'] * b['height'])
            yield (a['id'], b['id']), max(0.0, w) * max(0.0, h) / max(area, 1e-6)


def overprinted(source, design) -> bool:
    before = dict(_pairs(source))
    return any(share > TEXT_OVERPRINT and share > before.get(key, 0.0) + 0.05 for key, share in _pairs(design))


def failed_checks(report) -> int:
    return sum(not c['passed'] for c in report['checks'] if c['severity'] == 'error')


def vanished(source, design) -> int:
    """Texts the source shows that this layout draws too small to see."""
    out = {e['id']: e for e in design['elements']}
    shown = (e for e in source['elements'] if e['type'] == 'text' and str(e.get('content') or '').strip()
             and e.get('opacity', 1) > 0)
    return sum(1 for s in shown if (e := out.get(s['id'])) is None or e['font_size'] < MIN_TEXT_PX
               or min(e['width'], e['height']) < MIN_TEXT_PX)


def real_problem(source, report) -> bool:
    from engine.pipeline import _new_occlusion
    defects = report['quality']['defects']
    return bool(defects.get('overflow') or defects.get('contrast_regressions')
                or overprinted(source, report['design']) or _new_occlusion(source, report['design']))


def gate(source, pool):
    fewest = min(failed_checks(r) for r in pool)
    legal = [r for r in pool if failed_checks(r) == fewest]
    least = min(vanished(source, r['design']) for r in legal)
    legal = [r for r in legal if vanished(source, r['design']) == least]
    return [r for r in legal if not real_problem(source, r)] or legal


def _shortlist(source, target):
    from engine.pipeline import generate
    from merged.resize import install, is_floorplan, rank_key
    install()
    try:
        results = generate(source, target, all_candidates=True, options={'refine_seconds': REFINE_SECONDS})
    except Exception:
        return []
    results.sort(key=lambda r: rank_key(source, r), reverse=True)
    picked = results[:SHORTLIST]
    plans = [r for r in results if is_floorplan(r) and r not in picked]
    return picked + plans[:2]


def _foreign(source, target, items):
    from engine.pipeline import analyze, assess_design
    pool = []
    for item in items:
        try:
            design = comply(source, {'canvas': target, 'elements': item['design']['elements']})
            report = assess_design(source, target, design, analyze(source), item['name'])
        except Exception:
            continue
        report['template'] = 'gaurav:' + item['name']
        pool.append(report)
    return pool


def _uniform(source, target):
    """Last resort, reached only if every generator failed: the source scaled to fit, centred."""
    k = min(target['width'] / source['canvas']['width'], target['height'] / source['canvas']['height'])
    dx, dy = (target['width'] - source['canvas']['width'] * k) / 2, (target['height'] - source['canvas']['height'] * k) / 2
    elements = []
    for s in source['elements']:
        e = copy.deepcopy(s)
        e.update(x=s['x'] * k + dx, y=s['y'] * k + dy, width=s['width'] * k, height=s['height'] * k)
        if e['type'] == 'text':
            e['font_size'] = max(1, round(s['font_size'] * k))
        elements.append(e)
    return comply(source, {'canvas': target, 'elements': elements})


def resize_with_report(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    target = {'width': int(target_canvas['width']), 'height': int(target_canvas['height'])}
    with ThreadPoolExecutor(1) as helper:   # his engine runs in its own process: ask it while ours works
        theirs = helper.submit(gaurav.candidates, source, target)
        ours = _shortlist(source, target)
        pool = ours + _foreign(source, target, theirs.result())
    if not pool:
        return {'design': _uniform(source, target), 'template': 'uniform', 'candidates': 0, 'failed_checks': None}
    survivors = gate(source, pool)
    scored = [(prefs.score(prefs.layout_features(source, r) + pixels.features(r['design'])), i, r)
              for i, r in enumerate(survivors)]
    _, _, top = max(scored, key=lambda t: (t[0], -t[1]))
    return {'design': copy.deepcopy(top['design']), 'template': top['template'], 'candidates': len(pool),
            'survivors': len(survivors), 'failed_checks': failed_checks(top)}


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    return resize_with_report(source, target_canvas)['design']
