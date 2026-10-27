"""Training data for the learned appeal scorer (plan section 3.6).

Each example is a resized design, judged against its source on the target canvas:
  label 1  the designer's own resize (Crello pair target, rebuilt with the source's layers)
  label 0  the same resize with one principle broken (training.perturb plus appeal perturbations:
           off-balance, crowding to an edge, under-fill, uneven spacing, cramped headline), and
           today's pipeline output for the pair when it carries a hard or appeal flag.
Only train-split pairs are used; val/test stay for validation and the human rounds.

  python work/postproc/appeal/build.py --pairs 3000 --out work/postproc/appeal/examples.jsonl
"""
import argparse
import copy
import json
import math
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402
from appeal.features import features  # noqa: E402

ENV = None


def init():
    global ENV
    import warnings
    warnings.filterwarnings('ignore')
    harness.init('methods:current')
    ENV = harness.ENV


def designer_design(src, D, B, mp):
    """The designer's resize with the source's own layers; unmapped elements per-axis stretched."""
    from training.retarget.visual import designer_reference
    ref = designer_reference(src, D, B, mp)
    by = {e['id']: e for e in ref['elements']}
    tW, tH = B['W'], B['H']
    els = []
    for e in src['elements']:
        o = copy.deepcopy(e)
        if e['id'] in by:
            g = by[e['id']]
            o.update(x=g['x'], y=g['y'], width=g['width'], height=g['height'], rotation=g.get('rotation', 0))
            if e['type'] == 'text':
                o['font_size'] = g['font_size']
        else:
            o.update(x=e['x'] * tW / D['W'], y=e['y'] * tH / D['H'], width=e['width'] * tW / D['W'], height=e['height'] * tH / D['H'])
        els.append(o)
    return {'canvas': {'width': int(tW), 'height': int(tH)}, 'elements': els}


def appeal_perturbations(design, A, rng):
    """Composition-level flaws the rubric perturbations do not cover."""
    cw, ch = design['canvas']['width'], design['canvas']['height']
    movable = [e for e in design['elements'] if not (e['width'] >= .95 * cw and e['height'] >= .95 * ch)]
    if not movable:
        return
    x0 = min(e['x'] for e in movable); y0 = min(e['y'] for e in movable)
    x1 = max(e['x'] + e['width'] for e in movable); y1 = max(e['y'] + e['height'] for e in movable)
    # off-balance: the whole composition slides toward one corner
    for sev, f in (('medium', .5), ('large', 1.0)):
        d = copy.deepcopy(design)
        dx = -x0 * f if rng.random() < .5 else (cw - x1) * f
        dy = -y0 * f if rng.random() < .5 else (ch - y1) * f
        for e in d['elements']:
            if e['id'] in {m['id'] for m in movable}:
                e['x'] += dx; e['y'] += dy
        yield 'balance', sev, d
    # under-fill: content shrinks about its centre
    for sev, k in (('medium', .75), ('large', .55)):
        d = copy.deepcopy(design)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for e in d['elements']:
            if e['id'] in {m['id'] for m in movable}:
                e['x'] = cx + (e['x'] - cx) * k; e['y'] = cy + (e['y'] - cy) * k
                e['width'] *= k; e['height'] *= k
                if e['type'] == 'text':
                    e['font_size'] = max(1, round(e['font_size'] * k))
        yield 'fill', sev, d
    texts = [e for e in design['elements'] if e['type'] == 'text']
    if len(texts) >= 2:
        # uneven spacing: copy blocks jitter vertically by different amounts
        d = copy.deepcopy(design)
        for e in d['elements']:
            if e['type'] == 'text':
                e['y'] = min(max(0, e['y'] + rng.uniform(-.06, .06) * ch), ch - e['height'])
        yield 'rhythm', 'medium', d
        # crowding: copy pushed against the nearest edge
        d = copy.deepcopy(design)
        for e in d['elements']:
            if e['type'] == 'text':
                gaps = {'l': e['x'], 'r': cw - e['x'] - e['width'], 't': e['y'], 'b': ch - e['y'] - e['height']}
                side = min(gaps, key=gaps.get)
                if side == 'l': e['x'] = 0
                elif side == 'r': e['x'] = cw - e['width']
                elif side == 't': e['y'] = 0
                else: e['y'] = ch - e['height']
        yield 'margin', 'large', d


def task(pi):
    from training.perturb import perturbations
    out = []
    try:
        p = ENV.pairs[pi]
        D, B = ENV.designs[p['src']], ENV.designs[p['tgt']]
        src, A, recs = ENV.source(p['src'])
        si = harness.source_info(src, A)
        target = {'width': int(B['W']), 'height': int(B['H'])}
        good = designer_design(src, D, B, p['map'])
        rng = random.Random(pi)
        out.append({'pair': pi, 'label': 1, 'kind': 'designer', 'f': features(src, good, A, si)})
        bad = [(k, s, d) for k, s, _, d in perturbations(src, good, A, rng)]
        bad += list(appeal_perturbations(good, A, rng))
        rng.shuffle(bad)
        for k, s, d in bad[:8]:
            out.append({'pair': pi, 'label': 0, 'kind': f'perturb:{k}:{s}', 'f': features(src, d, A, si)})
        from engine.retarget.propose import propose
        cur = dict(propose(src, target, A, k=1))['0']
        fc = features(src, cur, A, si)
        out.append({'pair': pi, 'label': None, 'kind': 'current', 'f': fc})
    except Exception as exc:
        out.append({'pair': pi, 'error': repr(exc)[:160], 'tb': traceback.format_exc()[-500:]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=3000)
    ap.add_argument('--out', default=str(HERE / 'examples.jsonl'))
    a = ap.parse_args()
    from training.retarget.data import load
    designs, pairs = load(ROOT / 'work/retarget-data')
    by_src = {}
    for i, p in enumerate(pairs):
        if designs[p['src']]['split'] == 'train' and len(designs[p['src']]['els']) <= 48:
            by_src.setdefault(p['src'], []).append(i)
    rng = random.Random(9)
    srcs = sorted(by_src)
    rng.shuffle(srcs)
    pis = [rng.choice(by_src[s]) for s in srcs[:a.pairs]]
    pis.sort(key=lambda i: pairs[i]['src'])
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l).get('pair') for l in open(a.out)}
    todo = [i for i in pis if i not in done]
    t0 = time.time()
    with Pool(8, initializer=init) as pool, open(a.out, 'a') as fh:
        for n, rows in enumerate(pool.imap_unordered(task, todo, chunksize=2), 1):
            for r in rows:
                fh.write(json.dumps(r, default=float) + '\n')
            if n % 100 == 0:
                print(f'{n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
