"""Synthetic extreme-strip training data (plan: extreme-strip track).

Crello has no tall strip beyond 1:3.75 and few wide ones, so the model never learned them. For
train-split designs this builds strip layouts with the reading-order composer refined by the
constrained solver, keeps only those that pass every engine check and score well on the learned
appeal model, and stores them as target layouts in the training set's compact format, ready to be
added as extra (source, target) pairs for a model update.

  python work/postproc/strips/synth.py --designs 3000 --out work/postproc/strips/synthetic.jsonl
"""
import argparse
import json
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

FORMATS = ['800x90', '970x90', '728x90', '468x60', '90x800', '120x600', '160x600', '300x1050']
ENV = None


def init():
    global ENV
    import warnings
    warnings.filterwarnings('ignore')
    harness.init('methods:current')
    ENV = harness.ENV


def compact_target(design, src_compact, W, H):
    """Target layout as the dataset's compact element list (same order and indices as the source)."""
    by = {e['id']: e for e in design['elements']}
    els = []
    for e in src_compact['els']:
        d = by.get(f"el_{e['i']}")
        if d is None:
            return None
        els.append(dict(e, x=float(d['x']), y=float(d['y']), w=float(d['width']), h=float(d['height']),
                        font_size=float(d.get('font_size') or 0.0) if e['type'] == 1 else 0.0))
    return {'W': W, 'H': H, 'els': els}


def task(item):
    sidx, fmt = item
    from engine.retarget import appeal as AP
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    from engine.retarget.strip import compose
    try:
        src, A, recs = ENV.source(sidx)
        w, h = map(int, fmt.split('x'))
        target = {'width': w, 'height': h}
        P = Problem(src, target, A, recs, harness_priors())
        hint = compose(P)
        Ws, Hs = float(src['canvas']['width']), float(src['canvas']['height'])
        boxes, probs, _ = predict_dist(recs, Ws, Hs, w, h, k=1)
        cands = [('composer', hint)]
        sol = solve(P, boxes[0], probs[0], time_limit=1.0, hint=hint, workers=1)
        if sol is not None and sol.design is not None:
            cands.append(('solver', sol.design))
        si = AP.source_info(src, A)
        best = None
        for tag, d in cands:
            if not AP.eligible(src, target, d, A)[0]:
                continue
            s = AP.score(src, d, A, si)
            if best is None or s > best[0]:
                best = (s, tag, d)
        if best is None:
            return {'src': sidx, 'fmt': fmt, 'kept': False, 'why': 'none_eligible'}
        tgt = compact_target(best[2], ENV.designs[sidx], float(w), float(h))
        return {'src': sidx, 'src_id': ENV.designs[sidx]['id'], 'fmt': fmt, 'kept': tgt is not None, 'appeal': round(best[0], 4),
                'method': best[1], 'target': tgt}
    except Exception as exc:
        return {'src': sidx, 'fmt': fmt, 'kept': False, 'why': repr(exc)[:160], 'tb': traceback.format_exc()[-300:]}


_PRI = None


def harness_priors():
    global _PRI
    if _PRI is None:
        _PRI = json.load(open(ROOT / 'engine/models/designer_priors.json'))
    return _PRI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--designs', type=int, default=3000)
    ap.add_argument('--out', default=str(HERE / 'synthetic.jsonl'))
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    from training.retarget.data import load
    designs, _ = load(ROOT / 'work/retarget-data')
    train = [i for i, d in enumerate(designs) if d['split'] == 'train' and 3 <= len(d['els']) <= 48]
    rng = random.Random(31)
    items = [(i, rng.choice(FORMATS)) for i in rng.sample(train, min(a.designs, len(train)))]
    done = set()
    if Path(a.out).exists():
        done = {(json.loads(l)['src'], json.loads(l)['fmt']) for l in open(a.out)}
    items = [x for x in items if tuple(x) not in done]
    items.sort()
    t0 = time.time()
    with Pool(a.workers, initializer=init) as pool, open(a.out, 'a') as fh:
        for n, r in enumerate(pool.imap_unordered(task, items, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n')
            if n % 100 == 0:
                print(f'{n}/{len(items)} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
