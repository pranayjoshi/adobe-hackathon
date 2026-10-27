"""Score retargeting against what the designer actually did, on held-out families only.

Methods: uniform fit (letterbox), per-axis stretch, model greedy, model best-of-k selected
by a rule check (no peeking at the reference), and the oracle best-of-k (upper bound only).

    .venv/bin/python -m training.retarget.evaluate --weights engine/models/retarget.npz
"""
import argparse
import collections
import json
import math
import random

import numpy as np

from engine.retarget.features import BACKGROUND, TEXT
from engine.retarget.infer import Model, predict
from training.retarget.data import load, pair_example


def box(e):
    return e['x'], e['y'], e['x'] + e['w'], e['y'] + e['h']


def iou(a, b):
    ax0, ay0, ax1, ay1 = box(a); bx0, by0, bx1, by1 = box(b)
    iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a['w'] * a['h'] + b['w'] * b['h'] - inter
    return inter / u if u > 0 else 0.0


def overlap_frac(a, b):
    ax0, ay0, ax1, ay1 = box(a); bx0, by0, bx1, by1 = box(b)
    iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(1e-6, min(a['w'] * a['h'], b['w'] * b['h']))


def is_cover(e, W, H):
    return e['type'] == BACKGROUND or (e['w'] >= 0.95 * W and e['h'] >= 0.95 * H)


def fit(src, W, H, tW, tH):
    k = min(tW / W, tH / H); ox, oy = (tW - W * k) / 2, (tH - H * k) / 2
    out = []
    for e in src:
        if is_cover(e, W, H):
            out.append(dict(e, x=e['x'] * tW / W, y=e['y'] * tH / H, w=e['w'] * tW / W, h=e['h'] * tH / H)); continue
        out.append(dict(e, x=e['x'] * k + ox, y=e['y'] * k + oy, w=e['w'] * k, h=e['h'] * k, font_size=(e.get('font_size') or 0) * k))
    return out


def stretch(src, W, H, tW, tH):
    sx, sy = tW / W, tH / H; s = math.sqrt(sx * sy)
    out = []
    for e in src:
        if is_cover(e, W, H):
            out.append(dict(e, x=e['x'] * sx, y=e['y'] * sy, w=e['w'] * sx, h=e['h'] * sy)); continue
        w, h = e['w'] * s, e['h'] * s
        # Keep inside the canvas when the stretch pushes a smaller element out.
        w, h = (w * min(1, tW / w), h * min(1, tW / w)) if w > tW else (w, h)
        w, h = (w * min(1, tH / h), h * min(1, tH / h)) if h > tH else (w, h)
        cx, cy = (e['x'] + e['w'] / 2) * sx, (e['y'] + e['h'] / 2) * sy
        out.append(dict(e, x=min(max(cx - w / 2, 0), tW - w), y=min(max(cy - h / 2, 0), tH - h), w=w, h=h,
                        font_size=(e.get('font_size') or 0) * s * (w / (e['w'] * s))))
    return out


def apply_pred(src, pred, W, H, tW, tH):
    """Model boxes with the hard shape rules the engine also applies: graphics keep their aspect ratio,
    full-canvas backgrounds cover the target."""
    out = []
    for e, p in zip(src, pred):
        if is_cover(e, W, H):
            out.append(dict(e, x=0.0, y=0.0, w=float(tW), h=float(tH))); continue
        if e['type'] == TEXT:
            out.append(dict(e, x=p['x'], y=p['y'], w=p['w'], h=p['h'], font_size=p['font_size'])); continue
        a = e['w'] / e['h']; area = p['w'] * p['h']
        w, h = math.sqrt(area * a), math.sqrt(area / a)
        cx, cy = p['x'] + p['w'] / 2, p['y'] + p['h'] / 2
        out.append(dict(e, x=cx - w / 2, y=cy - h / 2, w=w, h=h))
    return out


def rule_defects(layout, tW, tH):
    """Reference-free checks: text overlapping text, text leaving the canvas."""
    texts = [e for e in layout if e['type'] == TEXT]
    d = 0.0
    for i in range(len(texts)):
        a = texts[i]
        out = (max(0, -a['x']) + max(0, a['x'] + a['w'] - tW)) / max(a['w'], 1) + (max(0, -a['y']) + max(0, a['y'] + a['h'] - tH)) / max(a['h'], 1)
        d += min(1.0, out) * 2
        for b in texts[i + 1:]:
            d += overlap_frac(a, b)
    return d


def metrics(layout, ref, tW, tH):
    els = [(a, b) for a, b in zip(layout, ref) if not is_cover(b, tW, tH)]
    if not els:
        return None
    m = {
        'iou': np.mean([iou(a, b) for a, b in els]),
        'center_err': np.mean([abs((a['x'] + a['w'] / 2 - b['x'] - b['w'] / 2) / tW) / 2 + abs((a['y'] + a['h'] / 2 - b['y'] - b['h'] / 2) / tH) / 2 for a, b in els]),
        'size_err': np.mean([abs(math.log(max(a['w'] * a['h'], 1e-3) / max(b['w'] * b['h'], 1e-3))) for a, b in els]),
    }
    t = [(a, b) for a, b in els if b['type'] == TEXT and (a.get('font_size') or 0) > 0 and (b.get('font_size') or 0) > 0]
    if t:
        m['font_err'] = np.mean([abs(math.log(a['font_size'] / b['font_size'])) for a, b in t])
    # Pairwise vertical/horizontal order agreement among non-background elements.
    agree = tot = 0
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            for axis, size in (('y', 'h'), ('x', 'w')):
                ra = (els[i][1][axis] + els[i][1][size] / 2) - (els[j][1][axis] + els[j][1][size] / 2)
                if abs(ra) < 0.05 * (tH if axis == 'y' else tW):
                    continue
                pa = (els[i][0][axis] + els[i][0][size] / 2) - (els[j][0][axis] + els[j][0][size] / 2)
                agree += (ra > 0) == (pa > 0); tot += 1
    if tot:
        m['order_agree'] = agree / tot
    m['defects'] = rule_defects(layout, tW, tH)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='engine/models/retarget.npz')
    ap.add_argument('--data', default='work/retarget-data')
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--per-family', type=int, default=6)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    designs, pairs = load(a.data)
    model = Model(a.weights)
    rng = random.Random(0)
    ps = [p for p in pairs if designs[p['src']]['split'] == a.split]
    rng.shuffle(ps)
    per = collections.Counter(); chosen = []
    for p in ps:
        if per[p['family']] < a.per_family:
            chosen.append(p); per[p['family']] += 1
    chosen = chosen[:a.limit]
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, p in enumerate(chosen):
        src, (W, H), ref, (tW, tH), _ = pair_example(designs, p, False)
        preds = predict(src, W, H, tW, tH, k=a.k, seed=n, model=model)
        cands = [apply_pred(src, pr, W, H, tW, tH) for pr in preds]
        scored = [(metrics(c, ref, tW, tH), c) for c in cands]
        if scored[0][0] is None:
            continue
        sel = min(range(len(cands)), key=lambda i: (round(scored[i][0]['defects'], 2), i))
        oracle = max(range(len(cands)), key=lambda i: scored[i][0]['iou'])
        rows = {'fit': metrics(fit(src, W, H, tW, tH), ref, tW, tH),
                'stretch': metrics(stretch(src, W, H, tW, tH), ref, tW, tH),
                'model_greedy': scored[0][0], 'model_selected': scored[sel][0], 'oracle_best_of_k': scored[oracle][0],
                'designer': metrics(ref, ref, tW, tH)}
        bucket = 'extreme' if abs(math.log((W / H) / (tW / tH))) > math.log(3) else 'moderate'
        for name, m in rows.items():
            for k, v in m.items():
                res[name][k].append(v); res[name + '@' + bucket][k].append(v)
    summary = {name: {k: round(float(np.mean(v)), 4) for k, v in ms.items()} | {'n': len(ms['iou'])} for name, ms in res.items()}
    print(json.dumps(summary, indent=1))
    if a.out:
        open(a.out, 'w').write(json.dumps({'pairs': len(chosen), 'families': len(per), 'summary': summary}, indent=1))


if __name__ == '__main__':
    main()
