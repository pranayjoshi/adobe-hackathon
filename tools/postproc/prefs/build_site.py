"""Build the voting site's manifest and images: python work/postproc/prefs/build_site.py r0 [r1 ...]

Round folders hold <key>_<arm>.jpg (+ <key>_source.jpg). Each round lists its two arms; the page
shows them in a per-viewer random order and stores the winning arm, never a side."""
import json, sys
from pathlib import Path
from PIL import Image

ROOT = Path('work/postproc/prefs')
SITE = ROOT / 'site'
ROUNDS = {
    'r0': {'name': 'Round 0', 'question': 'Which resize looks better?', 'arms': ['engine', 'designer'],
           'note': "Today's engine against the designer's own resize."},
    'r1': {'name': 'Round 1', 'question': 'Which resize looks better?', 'arms': ['new', 'engine']},
    'r2': {'name': 'Round 2', 'question': 'Which resize looks better?', 'arms': ['new', 'engine']},
    'r2d': {'name': 'Round 2 · designer', 'question': 'Which resize looks better?', 'arms': ['new', 'designer']},
    'rs': {'name': 'Strips', 'question': 'Which strip looks better?', 'arms': ['new', 'engine']},
}


def main(names):
    manifest = {'rounds': []}
    if (SITE / 'manifest.json').exists():
        old = json.loads((SITE / 'manifest.json').read_text())
        manifest['rounds'] = [r for r in old['rounds'] if r['id'] not in names]
    for rid in names:
        cfg = ROUNDS[rid]
        src_dir = ROOT / (rid if rid != 'r2d' else 'r2')
        out = SITE / 'img' / rid
        out.mkdir(parents=True, exist_ok=True)
        keys = sorted({p.name.rsplit('_', 1)[0] for p in src_dir.glob('*_source.jpg')}
                      | ({p.name.rsplit('_', 1)[0] for p in src_dir.glob('*_engine.jpg')} if rid == 'rs' else set()))
        comps = []
        for k in keys:
            files = {a: src_dir / f'{k}_{a}.jpg' for a in cfg['arms'] + ['source']}
            if not all(files[a].exists() for a in cfg['arms']):
                continue
            item = {'id': f'{rid}:{k}', 'arms': {}}
            for a, f in files.items():
                if not f.exists():
                    continue
                im = Image.open(f).convert('RGB')
                im.thumbnail((720, 720))
                dst = out / f'{k}_{a}.jpg'
                im.save(dst, quality=80)
                rel = f'img/{rid}/{k}_{a}.jpg'
                if a == 'source':
                    item['source'] = rel
                    item['source_size'] = list(Image.open(f).size)
                else:
                    item['arms'][a] = rel
            meta = src_dir / f'{k}_{cfg["arms"][0]}.json'
            if meta.exists():
                t = json.loads(meta.read_text()).get('target')
                if t:
                    item['target'] = [t['width'], t['height']]
            comps.append(item)
        manifest['rounds'].append({'id': rid, **cfg, 'comparisons': comps})
        print(rid, len(comps), 'comparisons')
    (SITE / 'manifest.json').write_text(json.dumps(manifest))


if __name__ == '__main__':
    main(sys.argv[1:])
