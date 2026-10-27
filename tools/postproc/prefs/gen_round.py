"""Render the arms of a preference round: python work/postproc/prefs/gen_round.py --round r0 --split test --arms engine
Arms: engine (today's committed engine: generate with learned layouts), designer (Crello's own preview),
new (engine with CHAMELEON_POSTPROC=solver). Writes work/postproc/prefs/<round>/<pair>_<arm>.jpg + meta.json."""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, 'work/postproc')
OUT = Path('work/postproc/prefs')
ARGS = {}


def init(arm):
    import warnings; warnings.filterwarnings('ignore')
    if arm == 'new':
        os.environ['CHAMELEON_POSTPROC'] = 'solver'
    import harness
    harness.init('methods:current')
    ARGS['E'] = harness.ENV


def job(a):
    pi, arm, rnd, fmt = a
    from engine.pipeline import generate
    from training.retarget.visual import render
    from PIL import Image
    E = ARGS['E']
    try:
        p = E.pairs[pi] if pi >= 0 else None
        sidx = p['src'] if p else -pi - 1
        src, A, recs = E.source(sidx)
        if fmt:
            w, h = map(int, fmt.split('x')); target = {'width': w, 'height': h}
        else:
            B = E.designs[p['tgt']]; target = {'width': int(B['W']), 'height': int(B['H'])}
        key = f'{pi}' if not fmt else f'{sidx}_{fmt}'
        d = OUT / rnd; d.mkdir(parents=True, exist_ok=True)
        if (d / f'{key}_{arm}.jpg').exists() and (d / f'{key}_{arm}.json').exists():
            return json.loads((d / f'{key}_{arm}.json').read_text())
        t0 = time.time()
        top = generate(src, target, top_k=1, options={'learned_retarget': True})[0]
        png = d / f'{key}_{arm}.png'
        im = render(top['design'], str(png), size=900)
        im.convert('RGB').save(d / f'{key}_{arm}.jpg', quality=85); png.unlink(missing_ok=True)
        meta = {'key': key, 'arm': arm, 'template': top['template'], 'valid': top['valid'], 'seconds': round(time.time() - t0, 1), 'target': target}
        (d / f'{key}_{arm}.json').write_text(json.dumps(meta))
        return meta
    except Exception as exc:
        return {'key': str(pi), 'arm': arm, 'error': repr(exc)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--arm', default='engine')
    ap.add_argument('--strips', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.strips:
        items = [(-i - 1, a.arm, a.round, fmt) for i, fmt in json.load(open(OUT / 'strip_round.json'))]
    else:
        items = [(pi, a.arm, a.round, None) for pi in json.load(open(OUT / 'rounds.json'))[a.split]]
    with Pool(a.workers, initializer=init, initargs=(a.arm,)) as pool:
        meta = pool.map(job, items, chunksize=1)
    (OUT / a.round / f'meta_{a.arm}.json').write_text(json.dumps(meta, indent=1))
    print('done', sum('error' not in m for m in meta), 'of', len(meta))


if __name__ == '__main__':
    main()
