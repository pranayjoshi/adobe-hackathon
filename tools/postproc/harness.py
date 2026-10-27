"""Gate harness for the post-processing plan (docs/retarget/post-processing-plan.md).

Fixed subsets (frozen in subsets.json):
  ho    2,000 held-out designer pairs (the scale run's ablation subset, seed 1)
  panel 1,000 held-out designs x 3 panel targets (largest / ~0.7 / ~0.3 log-aspect change)
  tune  1,000 train designs x 2 targets (one pair target if any, one panel target) for tuning

Every output is scored with the engine's own eligibility (assess_design + _learned_eligible), the
scale run's reference-free checks, designer agreement where a designer pair exists, and (once
trained) the appeal score.

  python work/postproc/harness.py --method current --subset ho --out work/postproc/runs/current-ho.jsonl
"""
import argparse
import importlib
import json
import math
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'work' / 'postproc'  # frozen subsets and runs (git-ignored data)
ROOT = HERE.parents[1]  # repo root
sys.path.insert(0, str(HERE))  # scale_common, scale_run
from scale_common import Env, check, source_info, free_targets, ref_records, score_vs_ref, PANEL  # noqa: E402
import scale_run as R  # noqa: E402

ENV = None
METHOD = None


def subsets():
    p = OUT / 'subsets.json'
    if p.exists():
        return json.loads(p.read_text())
    env = Env()
    rng = random.Random(1)
    # held-out pairs: identical to the scale run's ablation subset
    ok = sorted({json.loads(l)['pair'] for l in open(ROOT / 'work/retarget-scale/runs_ref.jsonl') if json.loads(l)['split'] != 'train'})
    rng.shuffle(ok)
    ho = ok[:2000]
    held = [i for i, d in enumerate(env.designs) if d['split'] in ('val', 'test') and len(d['els']) <= 48]
    train = [i for i, d in enumerate(env.designs) if d['split'] == 'train' and len(d['els']) <= 48]
    r2 = random.Random(2)
    panel = []
    for i in r2.sample(held, 1000):
        d = env.designs[i]
        panel += [[i, k] for k, _, _ in free_targets(d['W'], d['H'])]
    by_src = {}
    for pi, pr in enumerate(env.pairs):
        if env.designs[pr['src']]['split'] == 'train':
            by_src.setdefault(pr['src'], []).append(pi)
    tune = []
    for i in random.Random(3).sample(train, 1000):
        d = env.designs[i]
        if i in by_src:
            tune.append(['pair', by_src[i][0]])
        k = free_targets(d['W'], d['H'])[0][0]
        tune.append(['panel', i, k])
    out = {'ho': ho, 'panel': panel, 'tune': tune}
    p.write_text(json.dumps(out))
    return out


def strip_subset():
    """300 held-out designs x 2 extreme strips (one wide, one tall), frozen in strips.json."""
    p = OUT / 'strips.json'
    if p.exists():
        return json.loads(p.read_text())
    from training.retarget.data import load
    designs, _ = load(ROOT / 'work/retarget-data')
    held = [i for i, d in enumerate(designs) if d['split'] in ('val', 'test') and len(d['els']) <= 48]
    rng = random.Random(5)
    wide, tall = ['800x90', '970x90', '728x90'], ['90x800', '120x600', '160x600', '300x1050']
    out = [[i, rng.choice(wide)] for i in rng.sample(held, 300)]
    out = [x for i, w in out for x in ([i, w], [i, rng.choice(tall)])]
    p.write_text(json.dumps(out))
    return out


def eligible(src, target, design, A):
    """Exactly what the engine checks before it accepts a learned layout."""
    from engine.pipeline import assess_design, _learned_eligible, _content_share, _new_occlusion
    from engine.refine import blocking
    r = assess_design(src, target, design, A)
    reasons = []
    if not r['valid']:
        reasons.append('invalid:' + ','.join(c['name'] for c in r['checks'] if not c['passed'] and c['severity'] == 'error'))
    if not r.get('rubric', {}).get('gate'):
        reasons.append('rubric_gate')
    try:
        if blocking(r):
            reasons.append('blocking')
    except Exception:
        pass
    for k, v in r['quality']['defects'].items():
        if v and k != 'scene_displacement':
            reasons.append('defect:' + k)
    if r['diagnostics'].get('collisions'):
        reasons.append('collisions')
    if r['diagnostics'].get('contrast_regressions'):
        reasons.append('contrast')
    from engine.pipeline import _collapse_floor
    if _content_share(design) < _collapse_floor(design) * _content_share(src):
        reasons.append('collapse')
    if _new_occlusion(src, design):
        reasons.append('occlusion')
    ok = _learned_eligible(r, src)
    tier2 = (r['rubric'].get('tier2') or {})
    return ok, sorted(set(reasons)), {'overall': tier2.get('overall'), 'scores': tier2.get('scores')}


def init(method):
    global ENV, METHOD
    import warnings
    warnings.filterwarnings('ignore')
    ENV = Env()
    mod, fn = method.split(':') if ':' in method else ('methods', method)
    METHOD = getattr(importlib.import_module(mod), fn)


def _emit(kind, key, D, tW, tH, src, A, si, design, info, ref=None, mp=None, keys=None, dt=0.0):
    from training.retarget.visual import as_records
    target = {'width': int(tW), 'height': int(tH)}
    row = dict(kind=kind, key=key, **R.meta(D, tW, tH), t=round(dt, 3), info=info)
    if design is None:
        row['failed'] = True
        return row
    ok, reasons, rub = eligible(src, target, design, A)
    row.update(eligible=ok, veto=reasons, rubric=rub, chk=check(src, design, A, si, None))
    if ref is not None:
        row['m'] = score_vs_ref(as_records(design, src, D, mp), ref, tW, tH, keys, extra=False)
    try:
        from appeal import score as appeal_score
        row['appeal'] = appeal_score(src, design, A)
    except Exception:
        pass
    row['design'] = design
    return row


def task(item):
    kind = item[0]
    try:
        if kind == 'pair':
            pi = item[1]
            p = ENV.pairs[pi]
            sidx = p['src']
            D, B = ENV.designs[sidx], ENV.designs[p['tgt']]
            tW, tH = B['W'], B['H']
        else:
            sidx, fmt = item[1], item[2]
            D = ENV.designs[sidx]
            tW, tH = PANEL.get(fmt) or tuple(int(v) for v in fmt.split('x'))
        src, A, recs = ENV.source(sidx)
        si = source_info(src, A)
        target = {'width': int(tW), 'height': int(tH)}
        t0 = time.time()
        design, info = METHOD(src, target, A, recs)
        dt = time.time() - t0
        if kind == 'pair':
            mp = p['map']
            by = {e['id']: e for e in src['elements']}
            keys = [(by[f"el_{D['els'][i]['i']}"]['type'], A.roles.get(f"el_{D['els'][i]['i']}", '?')) for i in sorted(mp)]
            return _emit(kind, pi, D, tW, tH, src, A, si, design, info, ref_records(B, mp), mp, keys, dt)
        return _emit(kind, f'{sidx}:{fmt}', D, tW, tH, src, A, si, design, info, dt=dt)
    except Exception as exc:
        return {'kind': kind, 'key': str(item[1:]), 'error': repr(exc)[:200], 'tb': traceback.format_exc()[-600:]}


def items(name, limit=None):
    s = subsets()
    if name == 'ho':
        it = [['pair', pi] for pi in s['ho']]
    elif name == 'panel':
        it = [['panel', i, k] for i, k in s['panel']]
    elif name == 'tune':
        it = s['tune']
    elif name == 'strip':
        it = [['panel', i, k] for i, k in strip_subset()]
    else:
        raise ValueError(name)
    return it[:limit] if limit else it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--subset', default='ho')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-designs', action='store_true')
    a = ap.parse_args()
    it = items(a.subset, a.limit)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            done.add(json.dumps(r.get('key')))
    todo = [i for i in it if json.dumps(i[1] if i[0] == 'pair' else f'{i[1]}:{i[2]}') not in done]
    # group by source design so each worker reuses its converted source
    todo.sort(key=lambda i: ENV_SRC(i))
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.method,)) as pool, open(out, 'a') as fh:
        for n, row in enumerate(pool.imap(task, todo, chunksize=4), 1):
            if not a.keep_designs:
                row.pop('design', None)
            fh.write(json.dumps(row, default=float) + '\n')
            if n % 200 == 0:
                print(f'{n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    print('done', len(todo), f'{time.time() - t0:.0f}s')


_PAIRS = None


def ENV_SRC(i):
    global _PAIRS
    if i[0] != 'pair':
        return i[1]
    if _PAIRS is None:
        import pickle
        _PAIRS = pickle.load(open(ROOT / 'work/retarget-data/pairs.pkl', 'rb'))
    return _PAIRS[i[1]]['src']


if __name__ == '__main__':
    main()
