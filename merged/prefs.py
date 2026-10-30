"""What the judges prefer, learned from the blind study's votes.

`features(source, design, report)` measures a finished layout: geometry from its JSON alone,
plus the defect counts of our engine's checks (`report`, from `engine.pipeline.assess_design`
or a pipeline candidate), so it scores any engine's output the same way. `score(source, design)`
is a linear model over those features, fitted by `tools/blind_eval/fit_prefs.py` on the
pairwise preferences in the votes (winner beats each other option shown) and saved in
`prefs_weights.json`. Higher is better; only differences between layouts of one task mean
anything.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

WEIGHTS = Path(__file__).with_name('prefs_weights.json')

NAMES = ('headline', 'headline_keep', 'text_min', 'coverage', 'span', 'letterbox',
         'stretch', 'offcanvas', 'new_overlap', 'missing', 'text_clipped',
         'defects', 'collisions', 'unreadable', 'invalid')


def _box(e):
    return e['x'], e['y'], e['x'] + e['width'], e['y'] + e['height']


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _texts(d):
    return [e for e in d['elements'] if e['type'] == 'text' and (e.get('content') or '').strip()
            and e.get('opacity', 1) > 0]


def _backgrounds(d):
    W, H = d['canvas']['width'], d['canvas']['height']
    return [e for e in d['elements'] if e['type'] != 'text' and e['width'] * e['height'] >= 0.85 * W * H]


def _overlap(d):
    t = [_box(e) for e in _texts(d)]
    W, H = d['canvas']['width'], d['canvas']['height']
    return sum(_inter(a, b) for i, a in enumerate(t) for b in t[i + 1:]) / (W * H)


def checks(report) -> list[float]:
    q = report['quality']
    t2 = ((report.get('rubric') or {}).get('tier2') or {}).get('defects', [])
    d = q['defects']
    return [math.log1p(q['defect_count'] + len(t2)), math.log1p(len(d.get('collisions', []))),
            math.log1p(len(d.get('unreadable', []))), float(not report['valid'])]


def features(source: dict[str, Any], design: dict[str, Any], report: dict[str, Any]) -> list[float]:
    W, H = float(design['canvas']['width']), float(design['canvas']['height'])
    sW, sH = float(source['canvas']['width']), float(source['canvas']['height'])
    short, s_short = min(W, H), min(sW, sH)
    src = {e['id']: e for e in source['elements']}
    out = {e['id']: e for e in design['elements']}
    texts = _texts(design)
    src_texts = _texts(source)

    head_id = max(src_texts, key=lambda e: e['font_size'])['id'] if src_texts else None
    head = out.get(head_id)
    if head and head_id:
        rel = head['font_size'] / short
        headline = math.log(max(rel, 1e-3))
        headline_keep = math.log(max(rel / (src[head_id]['font_size'] / s_short), 1e-3))
    else:
        headline = headline_keep = 0.0
    text_min = math.log(max(min((e['font_size'] for e in texts), default=12.0), 1.0))

    bg_ids = {e['id'] for e in _backgrounds(source)}
    content = [e for e in design['elements'] if e['id'] not in bg_ids and e.get('opacity', 1) > 0]
    clip = lambda b: (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
    boxes = [clip(_box(e)) for e in content]
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if boxes:
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        coverage = (x1 - x0) * (y1 - y0) / (W * H)
        span = min((x1 - x0) / W, (y1 - y0) / H)
    else:
        coverage, span = 0.0, 0.0
    bgs = [out[i] for i in bg_ids if i in out]
    letterbox = 0.0
    if bgs:
        letterbox = 1.0 - max(_inter(clip(_box(e)), (0, 0, W, H)) for e in bgs) / (W * H)

    stretch = 0.0
    for e in design['elements']:
        s = src.get(e['id'])
        if e['type'] == 'image' and e.get('fit') != 'cover' and s and s['height'] > 0 and e['height'] > 0:  # cover crops, never distorts
            stretch = max(stretch, abs(math.log((e['width'] / e['height']) / (s['width'] / s['height']))))
    area = sum(e['width'] * e['height'] for e in content) or 1.0
    offcanvas = sum(e['width'] * e['height'] - _inter(_box(e), (0, 0, W, H)) for e in content) / area
    new_overlap = max(0.0, _overlap(design) - _overlap(source))
    shown = [e for e in source['elements'] if e.get('opacity', 1) > 0]
    missing = sum(1 for e in shown if e['id'] not in out or out[e['id']].get('opacity', 1) <= 0) / max(len(shown), 1)
    text_clipped = sum(1 for e in texts if _inter(_box(e), (0, 0, W, H)) < 0.98 * e['width'] * e['height']) / max(len(texts), 1)
    return [headline, headline_keep, text_min, coverage, span, letterbox,
            min(stretch, 1.0), offcanvas, new_overlap * 10, missing, text_clipped, *checks(report)]


_model = None


def score(source: dict[str, Any], design: dict[str, Any], report: dict[str, Any]) -> float:
    global _model
    if _model is None:
        _model = json.loads(WEIGHTS.read_text())
    f = features(source, design, report)
    m, s, w = _model['mean'], _model['scale'], _model['weights']
    return sum(wi * (fi - mi) / si for wi, fi, mi, si in zip(w, f, m, s))
