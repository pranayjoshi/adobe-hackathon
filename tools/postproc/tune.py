"""Tune the solver's soft-cost weights for appeal (plan P5) on train-split pairs.

  PYTHONPATH=. .venv/bin/python tools/postproc/tune.py --n 80 --configs 12 --out work/postproc/runs/tune.json

Each config multiplies the default weights (engine/retarget/solver.py WEIGHTS) by log-uniform
factors in [1/3, 3]. A config is scored on the same pairs by the engine's own pass/fail check and
the appeal scorer: mean appeal, with failing layouts charged 0.3. Config 0 is the defaults.
Only train-split pairs are used; held-out and voting designs never enter tuning.
"""
import argparse
import json
import random
import statistics as S
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

KEYS = ['model', 'align', 'margin', 'balance', 'rhythm', 'typo', 'headline', 'host', 'order_flip']
PENALTY = 0.3
W = {}


def init(tl):
    import warnings
    warnings.filterwarnings('ignore')
    harness.init('methods:current')
    W['tl'] = tl


def job(arg):
    ci, weights, pi = arg
    from engine.retarget import appeal as AP
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    from engine.retarget.propose import propose, _priors
    E = harness.ENV
    p = E.pairs[pi]
    src, A, recs = E.source(p['src'])
    B = E.designs[p['tgt']]
    t = {'width': int(B['W']), 'height': int(B['H'])}
    warm = dict(propose(src, t, A, k=1))['0']
    boxes, probs, _ = predict_dist(recs, float(src['canvas']['width']), float(src['canvas']['height']), t['width'], t['height'], k=1)
    sol = solve(Problem(src, t, A, recs, _priors()), boxes[0], probs[0], weights=weights, time_limit=W['tl'], hint=warm, workers=1)
    d = sol.design if sol is not None and sol.design is not None else warm
    ok = AP.eligible(src, t, d, A)[0]
    a = AP.score(src, d, A)
    return ci, pi, bool(ok), float(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--configs', type=int, default=12)
    ap.add_argument('--tl', type=float, default=1.2)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from engine.retarget.solver import WEIGHTS
    pis = [x[1] for x in harness.subsets()['tune'] if x[0] == 'pair'][:a.n]
    rng = random.Random(7)
    configs = [{}]
    for _ in range(a.configs - 1):
        configs.append({k: round(WEIGHTS[k] * 3 ** rng.uniform(-1, 1), 4) for k in KEYS})
    jobs = [(ci, c, pi) for ci, c in enumerate(configs) for pi in pis]
    t0 = time.time()
    res = {}
    with Pool(a.workers, initializer=init, initargs=(a.tl,)) as pool:
        for k, (ci, pi, ok, sc) in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
            res.setdefault(ci, []).append((pi, ok, sc))
            if k % 50 == 0:
                print(f'{k}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
    summary = []
    for ci, c in enumerate(configs):
        rows = res.get(ci, [])
        if not rows:
            continue
        obj = S.mean(sc - (0 if ok else PENALTY) for _, ok, sc in rows)
        summary.append({'config': ci, 'weights': c, 'n': len(rows), 'objective': round(obj, 4),
                        'eligible': round(sum(ok for _, ok, _ in rows) / len(rows), 3),
                        'appeal': round(S.mean(sc for _, _, sc in rows), 4)})
    summary.sort(key=lambda s: -s['objective'])
    Path(a.out).write_text(json.dumps({'summary': summary, 'pairs': pis, 'raw': {str(k): v for k, v in res.items()}}, indent=1))
    for s in summary:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
