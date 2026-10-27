"""Source thumbnails for a round: Crello's preview of the source design when fetched, else our render.
  python work/postproc/prefs/prep_sources.py r1 --split val | python work/postproc/prefs/prep_sources.py rs --strips"""
import argparse, json, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, 'work/postproc')
OUT = Path('work/postproc/prefs')

ap = argparse.ArgumentParser(); ap.add_argument('round'); ap.add_argument('--split', default='val'); ap.add_argument('--strips', action='store_true')
a = ap.parse_args()
import harness
harness.init('methods:current'); E = harness.ENV
from training.retarget.visual import render
d = OUT / a.round; d.mkdir(parents=True, exist_ok=True)
if a.strips:
    items = [(f'{i}_{fmt}', i) for i, fmt in json.load(open(OUT / 'strip_round.json'))]
else:
    items = [(str(pi), E.pairs[pi]['src']) for pi in json.load(open(OUT / 'rounds.json'))[a.split]]
for key, sidx in items:
    dst = d / f'{key}_source.jpg'
    if dst.exists():
        continue
    pv = OUT / 'previews' / f"{E.designs[sidx]['id']}.png"
    if pv.exists():
        im = Image.open(pv)
    else:
        src, _, _ = E.source(sidx)
        png = d / f'{key}_source.png'; im = render(src, str(png), size=900); png.unlink(missing_ok=True)
    im.convert('RGB').save(dst, quality=85)
print('sources ok', len(items))
