"""Validate the appeal scorer on realistic held-out comparisons (not synthetic perturbations).

  PYTHONPATH=. .venv/bin/python tools/postproc/appeal/validate.py --n 200 --out work/postproc/runs/appeal-validate.json

For val-split designer pairs that were never used for training or tuning:
  designer > today        how often the scorer ranks the designer's own resize above today's output
  designer > today, by flags   the same, split by whether today's output fails an engine check
  solver vs today         how often the solver's layout scores above today's (a proxy, not proof)
Human votes, once there are enough, replace these as the scorer's real test (plan 3.6).
"""
import argparse
import json
import statistics as S
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402


def init():
    import warnings
    warnings.filterwarnings('ignore')
    harness.init('methods:current')


def job(pi):
    from engine.retarget import appeal as AP
    from engine.retarget.infer import predict_dist
    from engine.retarget.solver import Problem, solve
    from engine.retarget.propose import propose, _priors
    from appeal.build import designer_design
    E = harness.ENV
    try:
        p = E.pairs[pi]
        src, A, recs = E.source(p['src'])
        D, B = E.designs[p['src']], E.designs[p['tgt']]
        t = {'width': int(B['W']), 'height': int(B['H'])}
        si = AP.source_info(src, A)
        des = designer_design(src, D, B, p['map'])
        today = dict(propose(src, t, A, k=1))['0']
        boxes, probs, _ = predict_dist(recs, float(src['canvas']['width']), float(src['canvas']['height']), t['width'], t['height'], k=1)
        sol = solve(Problem(src, t, A, recs, _priors()), boxes[0], probs[0], time_limit=1.2, hint=today, workers=1)
        out = {'pair': pi}
        for name, d in (('designer', des), ('today', today), ('solver', sol.design if sol and sol.design else None)):
            if d is None:
                continue
            ok = AP.eligible(src, t, d, A)[0]
            out[name] = {'ok': bool(ok), 'appeal': AP.score(src, d, A, si)}
        return out
    except Exception as exc:
        return {'pair': pi, 'error': repr(exc)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from training.retarget.data import load
    designs, pairs = load(harness.ROOT / 'work' / 'retarget-data')
    sub = harness.subsets()
    used = set(sub['ho'])
    rounds = json.load(open(harness.ROOT / 'work' / 'postproc' / 'prefs' / 'rounds.json'))
    used |= {p for v in rounds.values() for p in v}
    import random
    cand = [i for i, p in enumerate(pairs) if designs[p['src']]['split'] == 'val' and i not in used and len(designs[p['src']]['els']) <= 48]
    random.Random(21).shuffle(cand)
    with Pool(a.workers, initializer=init) as pool:
        rows = pool.map(job, cand[:a.n], chunksize=1)
    ok = [r for r in rows if 'error' not in r and 'designer' in r and 'today' in r]
    res = {'n': len(ok), 'errors': len(rows) - len(ok)}
    res['designer_over_today'] = round(S.mean(r['designer']['appeal'] > r['today']['appeal'] for r in ok), 3)
    flag = [r for r in ok if not r['today']['ok']]
    clean = [r for r in ok if r['today']['ok']]
    res['designer_over_today_when_today_fails'] = (round(S.mean(r['designer']['appeal'] > r['today']['appeal'] for r in flag), 3), len(flag))
    res['designer_over_today_when_today_passes'] = (round(S.mean(r['designer']['appeal'] > r['today']['appeal'] for r in clean), 3), len(clean))
    both = [r for r in ok if 'solver' in r]
    res['solver_over_today'] = round(S.mean(r['solver']['appeal'] > r['today']['appeal'] for r in both), 3)
    res['mean_appeal'] = {k: round(S.mean(r[k]['appeal'] for r in ok if k in r), 4) for k in ('designer', 'today', 'solver')}
    res['pass_rate'] = {k: round(S.mean(r[k]['ok'] for r in ok if k in r), 3) for k in ('designer', 'today', 'solver')}
    Path(a.out).write_text(json.dumps({'summary': res, 'rows': rows}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
