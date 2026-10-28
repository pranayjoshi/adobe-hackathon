"""Render the designer arm with our own renderer: python tools/postproc/prefs/render_designer.py r2 --split test

The designer's resize geometry (from the Crello pair mapping) drawn with the source's layers by the
same renderer as the engine arms, so a vote compares layouts. Crello's own preview of the
designer's design uses its true fonts and effects, which would give that side away.
Writes work/postproc/prefs/<round>/<pair>_designer.jpg (900 px, like gen_round.py).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness  # noqa: E402

OUT = harness.ROOT / 'work' / 'postproc' / 'prefs'


def init():
    import warnings
    warnings.filterwarnings('ignore')
    harness.init('methods:current')


def job(arg):
    rnd, pi = arg
    from appeal.build import designer_design
    from training.retarget.visual import render
    E = harness.ENV
    try:
        p = E.pairs[pi]
        src, A, recs = E.source(p['src'])
        D, B = E.designs[p['src']], E.designs[p['tgt']]
        d = designer_design(src, D, B, p['map'])
        out = OUT / rnd
        out.mkdir(parents=True, exist_ok=True)
        png = out / f'{pi}_designer.png'
        im = render(d, str(png), size=900)
        im.convert('RGB').save(out / f'{pi}_designer.jpg', quality=85)
        png.unlink(missing_ok=True)
        (out / f'{pi}_designer.json').write_text(json.dumps({'key': str(pi), 'arm': 'designer', 'target': d['canvas']}))
        return pi, None
    except Exception as exc:
        return pi, repr(exc)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('round')
    ap.add_argument('--split', default='test')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    pis = json.load(open(OUT / 'rounds.json'))[a.split]
    with Pool(a.workers, initializer=init) as pool:
        res = pool.map(job, [(a.round, pi) for pi in pis], chunksize=1)
    bad = [r for r in res if r[1]]
    print('done', len(res) - len(bad), 'of', len(res), bad[:3])


if __name__ == '__main__':
    main()
