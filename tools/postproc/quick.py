"""Fast in-process evaluation of solver weight settings on a slice of the held-out subset (no rubric).

python work/postproc/quick.py --n 200 --set fill=0,balance=0
Reports designer IoU/font/size error and the engine's eligibility for the solver's design.
"""
import argparse, json, statistics as S, sys, time
from multiprocessing import Pool
sys.path.insert(0, 'work/postproc')
import harness

W = {}


def init(weights, tl, disable, current=False):
    harness.init('methods:current')
    W.update(weights=weights, tl=tl, disable=disable, current=current)


def job(pi):
    from engine.retarget.infer import predict_dist
    from engine.retarget import solver as SV
    from training.retarget.visual import as_records
    import methods
    E = harness.ENV
    p = E.pairs[pi]; src, A, recs = E.source(p['src']); D, B = E.designs[p['src']], E.designs[p['tgt']]
    t = {'width': int(B['W']), 'height': int(B['H'])}
    t0 = time.time()
    boxes, probs, _ = predict_dist(recs, src['canvas']['width'], src['canvas']['height'], t['width'], t['height'], k=1)
    P = SV.Problem(src, t, A, recs, methods._priors())
    from engine.retarget.propose import propose
    hint = dict(propose(src, t, A, k=1))['0']
    if W.get('current'):
        class _S: pass
        sol = _S(); sol.design, sol.parts, sol.status = hint, {}, 'current'
        return _finish(pi, src, A, D, B, p, t, sol, time.time() - t0)
    sol = SV.solve(P, boxes[0], probs[0], weights=W['weights'], time_limit=W['tl'], workers=1, rounds=6, disable=W['disable'], hint=hint)
    return _finish(pi, src, A, D, B, p, t, sol, time.time() - t0)


def _finish(pi, src, A, D, B, p, t, sol, dt):
    from training.retarget.visual import as_records
    if sol is None or sol.design is None:
        return {'pi': pi, 'fail': sol.status if sol else 'none', 't': dt}
    ok, reasons, _ = harness.eligible(src, t, sol.design, A)
    mp = p['map']; by = {e['id']: e for e in src['elements']}
    keys = [(by[f"el_{D['els'][i]['i']}"]['type'], A.roles.get(f"el_{D['els'][i]['i']}", '?')) for i in sorted(mp)]
    m = harness.score_vs_ref(as_records(sol.design, src, D, mp), harness.ref_records(B, mp), B['W'], B['H'], keys, extra=True)
    return {'pi': pi, 'ok': ok, 'veto': reasons, 'm': m, 't': dt, 'parts': sol.parts, 'status': sol.status, 'notes': getattr(sol, 'notes', [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--set', default='')
    ap.add_argument('--tl', type=float, default=1.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--disable', default='')
    ap.add_argument('--subset', default='ho')
    ap.add_argument('--current', action='store_true')
    a = ap.parse_args()
    weights = {k: float(v) for k, v in (kv.split('=') for kv in a.set.split(',') if kv)}
    sub = harness.subsets()
    pis = (sub['ho'] if a.subset == 'ho' else [x[1] for x in sub['tune'] if x[0] == 'pair'])[:a.n]
    rows = []
    t0 = time.time()
    with Pool(5, initializer=init, initargs=(weights, a.tl, tuple(x for x in a.disable.split(',') if x), a.current)) as pool:
        for k, r in enumerate(pool.imap_unordered(job, pis, chunksize=1), 1):
            rows.append(r)
            if k % 20 == 0:
                print(f'# {k}/{len(pis)} {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
    ok = [r for r in rows if 'm' in r and r['m']]
    import collections
    res = {'set': a.set + ' dis=' + a.disable, 'n': len(rows), 'fail': len(rows) - len(ok), 'eligible': round(sum(r['ok'] for r in ok) / len(rows), 3),
           'iou': round(S.mean(r['m']['iou'] for r in ok), 4), 'font_err': round(S.mean(r['m']['font_err'] for r in ok if r['m'].get('font_err') is not None), 4),
           'size_err': round(S.mean(r['m']['size_err'] for r in ok), 4), 't_med': round(S.median(r['t'] for r in rows), 2),
           'veto': collections.Counter(v.split(':')[0] + ':' + v.split(':')[1][:20] if ':' in v else v for r in ok for v in r['veto']).most_common(6),
           'fails': collections.Counter(r.get('fail') for r in rows if 'fail' in r).most_common(3)}
    print(json.dumps(res))
    if a.out:
        json.dump(rows, open(a.out, 'w'), default=float)


if __name__ == '__main__':
    main()
