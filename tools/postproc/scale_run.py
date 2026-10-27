"""Reference and reference-free runs.  python work/retarget-scale/run.py ref|free [--workers 8] [--limit N] [--ids FILE]"""
import argparse
import gzip
import json
import math
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_common import *  # noqa

ENV = None


def init():
    global ENV
    import warnings; warnings.filterwarnings('ignore')
    ENV = Env()


def meta(D, tW, tH):
    la = math.log((D['W'] / D['H']) / (tW / tH))
    return dict(src_id=D['id'], split=D['split'], family=D['family'], cat=D['category'], W=D['W'], H=D['H'], tW=tW, tH=tH,
                la=la, bucket=bucket(la), fmt=f'{int(tW)}x{int(tH)}')


def ref_task(task):
    """task = (src_idx, [pair_idx...], train_sel)"""
    from engine.retarget.infer import predict
    from training.retarget.visual import as_records
    from training.retarget.evaluate import fit as ev_fit, stretch as ev_stretch
    from training.retarget.data import pair_example
    sidx, pidxs, train_sel = task
    rows, skips = [], []
    D = ENV.designs[sidx]
    try:
        t0 = time.time()
        src, A, recs = ENV.source(sidx)
        t_an = time.time() - t0
        si = source_info(src, A)
    except ValueError as exc:
        return [], [{'kind': 'ref', 'src_id': D['id'], 'split': D['split'], 'reason': str(exc), 'pairs': len(pidxs)}]
    except Exception as exc:
        return [], [{'kind': 'ref', 'src_id': D['id'], 'split': D['split'], 'reason': 'exc:' + repr(exc)[:100], 'pairs': len(pidxs)}]
    for pi in pidxs:
        p = ENV.pairs[pi]
        Bd = ENV.designs[p['tgt']]
        tW, tH = Bd['W'], Bd['H']
        target = {'width': int(tW), 'height': int(tH)}
        row = dict(kind='ref', pair=pi, tgt_id=Bd['id'], train_sel=train_sel, **meta(D, tW, tH))
        try:
            mp = p['map']
            keys = [(src['elements'][0]['type'] and None, None)]  # placeholder replaced below
            by = {e['id']: e for e in src['elements']}
            keys = []
            for si_ in sorted(mp):
                eid = f"el_{D['els'][si_]['i']}"
                keys.append((by[eid]['type'], A.roles.get(eid, by[eid]['type'])))
            ref = ref_records(Bd, mp)
            row.update(n_el=len(src['elements']), n_text=sum(e['type'] == 'text' for e in src['elements']), n_mapped=len(mp),
                       n_cover_src=sum(r['cover'] for r in recs), t_analyze=t_an)
            t0 = time.time()
            props = ENV.P.propose(src, target, A, k=8)
            row['t_propose8'] = time.time() - t0
            if not props:
                skips.append(dict(kind='ref', src_id=D['id'], split=D['split'], reason='propose_empty', pairs=1)); continue
            preds = predict(recs, D['W'], D['H'], float(tW), float(tH), k=8, temperature=0.8, seed=0)
            raw = raw_design(src, target, recs, preds[0])
            pred_fs = {r['id']: pr['font_size'] for r, pr in zip(recs, preds[0]) if r['type'] == 1}
            dsg, _ = ENV.designs, None
            sr, (W, H), rf, _, _ = pair_example(ENV.designs, p, False)
            var = {}
            var['stretch'] = {'m': score_vs_ref(ev_stretch(sr, W, H, tW, tH), rf, tW, tH, [(None, None)] * len(sr), False)}
            var['fit'] = {'m': score_vs_ref(ev_fit(sr, W, H, tW, tH), rf, tW, tH, [(None, None)] * len(sr), False)}
            Ac = D

            def rec(design):
                return as_records(design, src, Ac, mp)
            var['raw'] = {'m': score_vs_ref(rec(raw), ref, tW, tH, keys)}
            samples = []
            best = None
            tagged = {}
            for tag, des in props:
                tagged[tag] = des
                m = score_vs_ref(rec(des), ref, tW, tH, keys, extra=False)
                if m:
                    samples.append(dict(tag=tag, iou=m['iou'], center=m['center_err'], defects=m['defects']))
                    if tag.isdigit() and (best is None or m['iou'] > best[0]):
                        best = (m['iou'], tag)
            if not samples:
                skips.append(dict(kind='ref', src_id=D['id'], split=D['split'], reason='no_scorable_elements', pairs=1)); continue
            row['samples'] = samples
            var['p0'] = {'m': score_vs_ref(rec(tagged['0']), ref, tW, tH, keys)}
            if '0_bound' in tagged:
                var['bound'] = {'m': score_vs_ref(rec(tagged['0_bound']), ref, tW, tH, keys)}
            row['oracle_tag'] = best[1]
            if best[1] != '0':
                var['oracle'] = {'m': score_vs_ref(rec(tagged[best[1]]), ref, tW, tH, keys)}
            t0 = time.time()
            var['raw']['chk'] = check(src, raw, A, si, pred_fs)
            var['p0']['chk'] = check(src, tagged['0'], A, si, pred_fs)
            if '0_bound' in tagged:
                var['bound']['chk'] = check(src, tagged['0_bound'], A, si, pred_fs)
            if best[1] != '0':
                var['oracle']['chk'] = check(src, tagged[best[1]], A, si, pred_fs)
            row['t_checks'] = time.time() - t0
            row['v'] = var
            rows.append(row)
        except Exception as exc:
            skips.append(dict(kind='ref', src_id=D['id'], split=D['split'], reason='run_exc:' + repr(exc)[:120], tb=traceback.format_exc()[-400:], pairs=1))
    return rows, skips


def free_task(task):
    from engine.retarget.infer import predict
    sidx = task
    D = ENV.designs[sidx]
    rows, skips = [], []
    try:
        t0 = time.time()
        src, A, recs = ENV.source(sidx)
        t_an = time.time() - t0
        si = source_info(src, A)
    except ValueError as exc:
        return [], [{'kind': 'free', 'src_id': D['id'], 'split': D['split'], 'reason': str(exc)}]
    except Exception as exc:
        return [], [{'kind': 'free', 'src_id': D['id'], 'split': D['split'], 'reason': 'exc:' + repr(exc)[:100]}]
    for name, d, (tW, tH) in free_targets(D['W'], D['H']):
        target = {'width': tW, 'height': tH}
        row = dict(kind='free', **meta(D, tW, tH))
        try:
            row.update(n_el=len(src['elements']), n_text=sum(e['type'] == 'text' for e in src['elements']), t_analyze=t_an)
            t0 = time.time()
            props = ENV.P.propose(src, target, A, k=1)
            row['t_propose1'] = time.time() - t0
            if not props:
                skips.append(dict(kind='free', src_id=D['id'], split=D['split'], reason='propose_empty', fmt=name)); continue
            preds = predict(recs, D['W'], D['H'], float(tW), float(tH), k=1, temperature=0.8, seed=0)
            raw = raw_design(src, target, recs, preds[0])
            pred_fs = {r['id']: pr['font_size'] for r, pr in zip(recs, preds[0]) if r['type'] == 1}
            t0 = time.time()
            var = {'raw': {'chk': check(src, raw, A, si, pred_fs)}}
            for tag, des in props:
                var['p0' if tag == '0' else 'bound'] = {'chk': check(src, des, A, si, pred_fs)}
            row['t_checks'] = time.time() - t0
            row['v'] = var
            rows.append(row)
        except Exception as exc:
            skips.append(dict(kind='free', src_id=D['id'], split=D['split'], reason='run_exc:' + repr(exc)[:120], tb=traceback.format_exc()[-400:], fmt=name))
    return rows, skips


def plan_ref(env):
    import collections
    pairs, designs = env.pairs, env.designs
    tasks = collections.defaultdict(list)
    best = {}
    for pi, p in enumerate(pairs):
        s = designs[p['src']]['split']
        if s in ('val', 'test'):
            tasks[p['src']].append(pi)
        else:
            la = abs(math.log((designs[p['src']]['W'] / designs[p['src']]['H']) / (designs[p['tgt']]['W'] / designs[p['tgt']]['H'])))
            if p['src'] not in best or la > best[p['src']][0]:
                best[p['src']] = (la, pi)
    out = [(s, v, False) for s, v in tasks.items()] + [(s, [b[1]], True) for s, b in best.items()]
    out.sort(key=lambda t: (t[2], t[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    env = Env()
    out = Path(a.out or SC / f'runs_{a.kind}.jsonl'); skp = out.with_suffix('.skips.jsonl')
    done = set()
    if out.exists():
        for l in open(out):
            if not l.strip(): continue
            r = json.loads(l); done.add(r['src_id'])
    if skp.exists():
        for l in open(skp):
            if not l.strip(): continue
            r = json.loads(l)
            if r['reason'] != 'no_image_file':
                done.add(r['src_id'])
    if a.kind == 'ref':
        tasks = plan_ref(env); fn = ref_task; sid = lambda t: env.designs[t[0]]['id']
    else:
        tasks = list(range(len(env.designs))); fn = free_task; sid = lambda t: env.designs[t]['id']
    tasks = [t for t in tasks if sid(t) not in done]
    if a.limit:
        tasks = tasks[:a.limit]
    print(len(tasks), 'tasks', flush=True)
    t0 = time.time(); n = 0
    with Pool(a.workers, initializer=init) as pool, open(out, 'a') as fo, open(skp, 'a') as fs:
        for rows, skips in pool.imap_unordered(fn, tasks, chunksize=4):
            for r in rows:
                fo.write(json.dumps(jsonable(r)) + '\n')
            for s in skips:
                fs.write(json.dumps(jsonable(s)) + '\n')
            n += 1
            if n % 50 == 0:
                fo.flush(); fs.flush()
                print(f'{n}/{len(tasks)} {time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0, flush=True)


if __name__ == '__main__':
    main()
