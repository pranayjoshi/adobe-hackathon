"""Build the round-2 voting site (test split): python tools/postproc/prefs/build_round2_site.py

Two comparisons per design, sharing one set of four images so the page stays under an artifact
version's 511-file limit:
  r2   new engine vs today's engine
  r2d  new engine vs the designer's own resize (drawn by our renderer: render_designer.py)
Inputs: work/postproc/prefs/r0/<pair>_{source,engine}.jpg (today's engine, rendered before the
solver became the default) and work/postproc/prefs/r2/<pair>_{new,designer}.jpg.
Image names are hashed so nothing in the page source reveals which side is which; the mapping is
kept in work/postproc/prefs/r2/opaque_names.json. Output: work/postproc/prefs/site3/.
"""
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'work' / 'postproc' / 'prefs'
S = P / 'site3'
SALT = 'r2-9c3e:'


def main():
    if S.exists():
        shutil.rmtree(S)
    (S / 'img').mkdir(parents=True)
    page = (Path(__file__).resolve().parent / 'site.html').read_text()
    (S / 'index.html').write_text(page.replace('<title>Resize Taste Test</title>', '<title>Final Resize Taste Test</title>', 1))
    src_of = {'source': P / 'r0', 'engine': P / 'r0', 'new': P / 'r2', 'designer': P / 'r2'}
    names = {}
    r2, r2d = [], []
    for pi in json.load(open(P / 'rounds.json'))['test']:
        files = {a: d / f'{pi}_{a}.jpg' for a, d in src_of.items()}
        if not all(f.exists() for f in files.values()):
            print('missing', pi)
            continue
        rel = {}
        for a, f in files.items():
            key = f'img/r2/{pi}_{a}.jpg'
            out = f'img/{hashlib.sha256((SALT + key).encode()).hexdigest()[:12]}.jpg'
            im = Image.open(f).convert('RGB')
            im.thumbnail((720, 720))
            im.save(S / out, quality=80)
            rel[a] = names[key] = out
        t = json.loads((P / 'r2' / f'{pi}_new.json').read_text())['target']
        base = {'source': rel['source'], 'source_size': list(Image.open(files['source']).size), 'target': [t['width'], t['height']]}
        r2.append({'id': f'r2:{pi}', 'arms': {'new': rel['new'], 'engine': rel['engine']}, **base})
        r2d.append({'id': f'r2d:{pi}', 'arms': {'new': rel['new'], 'designer': rel['designer']}, **base})
    manifest = {'rounds': [
        {'id': 'r2', 'name': 'Round 2', 'question': 'Which resize looks better?', 'arms': ['new', 'engine'], 'comparisons': r2},
        {'id': 'r2d', 'name': 'Round 2 · designer', 'question': 'Which resize looks better?', 'arms': ['new', 'designer'], 'comparisons': r2d}]}
    (S / 'manifest.json').write_text(json.dumps(manifest))
    (P / 'r2' / 'opaque_names.json').write_text(json.dumps(names, indent=0))
    print(len(r2), 'designs,', len(names), 'images')


if __name__ == '__main__':
    main()
