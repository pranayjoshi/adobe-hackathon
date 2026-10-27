"""Rendered check on held-out Crello families: source, designer resize, engine, engine+model, model only.

Also scores the engine's top candidate with and without the learned proposals against the
designer's own resize (same metrics as evaluate.py).

    .venv/bin/python -m training.retarget.visual --cases 50 --out output/retarget-visual
"""
import argparse
import collections
import copy
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from tools.build_crello_catalog import PrimitiveUnpickler, convert_row
from training.retarget.data import load
from training.retarget.evaluate import metrics

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'brand_kit/assets'
FONTS = ROOT / 'brand_kit/fonts'
FEATURES = json.load(open(ROOT / 'work/crello-cache/features.json'))


def raw_rows(ids):
    rows = {}
    for f in sorted((ROOT / 'work/retarget-data/raw').glob('*.parquet')):
        t = pq.read_table(f)
        for i, rid in enumerate(t['id'].to_pylist()):
            if rid in ids:
                rows[rid] = t.slice(i, 1).to_pylist()[0]
    return rows


def install_fonts(rows):
    from cr_renderer.fonts import normalize_family
    names = FEATURES['font']['feature']['names']
    wanted = {normalize_family(names[r['font'][i]]) for r in rows for i in range(r['length'])
              if r['type'][i] == 1 and r['font'][i] is not None}
    with open(ROOT / 'work/crello-cache/fonts.pickle', 'rb') as f:
        font_dict = PrimitiveUnpickler(f).load()
    (FONTS / 'crello').mkdir(parents=True, exist_ok=True)
    manifest = []
    for family, faces in font_dict.items():
        if normalize_family(family) not in wanted:
            continue
        for face in faces:
            name = hashlib.sha256(face['bytes']).hexdigest() + '.ttf'
            p = FONTS / 'crello' / name
            if not p.exists():
                p.write_bytes(face['bytes'])
            manifest.append({'file': 'crello/' + name, **{k: face.get(k, 'regular') for k in ('fontFamily', 'fontWeight', 'fontStyle')}})
    mp = FONTS / 'crello-fonts.json'
    old = json.loads(mp.read_text()) if mp.exists() else []
    merged = {(x['file'], x['fontFamily'], x['fontWeight'], x['fontStyle']): x for x in old + manifest}
    mp.write_text(json.dumps(list(merged.values()), indent=2))


def engine_source(row, images):
    row = dict(row, image=[{'bytes': b} if b else None for b in images['layers']])
    key, design, _ = convert_row(row, FEATURES, ASSETS)
    return key, design


def designer_reference(src_design, A, B, mapping):
    """The designer's resize, drawn with the source's own layers (matched elements only)."""
    out = {'canvas': {'width': int(B['W']), 'height': int(B['H'])}, 'elements': []}
    by_id = {e['id']: e for e in src_design['elements']}
    for si, ti in sorted(mapping.items(), key=lambda kv: B['els'][kv[1]]['i']):
        se, te = A['els'][si], B['els'][ti]
        e = copy.deepcopy(by_id[f"el_{se['i']}"])
        e.update(x=te['x'], y=te['y'], width=te['w'], height=te['h'], rotation=te['angle'], z_index=te['i'])
        if e['type'] == 'text':
            e['font_size'] = max(1.0, te['font_size'])
        out['elements'].append(e)
    return out


def as_records(design, src_design, A, mapping):
    """Engine design -> evaluate.py element dicts, in pair order, for scoring against the designer."""
    by_id = {e['id']: e for e in design['elements']}
    out = []
    for si in sorted(mapping):
        e = by_id[f"el_{A['els'][si]['i']}"]
        out.append({'type': A['els'][si]['type'], 'x': float(e['x']), 'y': float(e['y']), 'w': float(e['width']),
                    'h': float(e['height']), 'font_size': float(e.get('font_size') or 0)})
    return out


def render(design, path, size=360):
    from renderer.converter.convert import render_design_to_png
    render_design_to_png(design, ASSETS, path, max_size=size)
    return Image.open(path).convert('RGB')


def sheet(rows, labels, path, cell=300):
    pad, head = 10, 22
    W = len(labels) * (cell + pad) + pad
    H = len(rows) * (cell + head + pad) + pad + head
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    for c, lab in enumerate(labels):
        d.text((pad + c * (cell + pad), 4), lab, fill='black')
    for r, (title, images) in enumerate(rows):
        y = head + pad + r * (cell + head + pad)
        d.text((pad, y), title, fill='black')
        for c, img in enumerate(images):
            if img is None:
                continue
            t = img.copy(); t.thumbnail((cell, cell))
            x = pad + c * (cell + pad)
            d.rectangle([x - 1, y + head - 1, x + t.width, y + head + t.height], outline='#999')
            im.paste(t, (x, y + head))
    im.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=50)
    ap.add_argument('--out', default='output/retarget-visual')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--skip-engine', action='store_true')
    ap.add_argument('--methods', default='engine_before,engine_model,model_only')
    a = ap.parse_args()
    if a.weights:
        import os
        os.environ['CHAMELEON_RETARGET_WEIGHTS'] = str(Path(a.weights).resolve())
    from engine.analysis import analyze
    from engine.pipeline import generate
    from engine.retarget.propose import propose

    designs, pairs = load(ROOT / 'work/retarget-data')
    have = {p.stem for p in (ROOT / 'work/retarget-data/images').glob('*.pkl')}
    test = [p for p in pairs if designs[p['src']]['split'] == 'test' and designs[p['src']]['id'] in have]
    rng = random.Random(1)
    rng.shuffle(test)
    per_fam = collections.Counter(); cases = []
    for p in test:
        if per_fam[p['family']] < 1:
            cases.append(p); per_fam[p['family']] += 1
    cases = cases[:a.cases]
    rows = raw_rows({designs[p['src']]['id'] for p in cases})
    install_fonts(list(rows.values()))
    out = ROOT / a.out; (out / 'png').mkdir(parents=True, exist_ok=True)
    labels = ['source', 'designer resize', 'engine (before)', 'engine + model', 'model only (greedy)']
    sheet_rows, scores, log = [], collections.defaultdict(lambda: collections.defaultdict(list)), []
    for n, p in enumerate(cases):
        A, B = designs[p['src']], designs[p['tgt']]
        images = pickle.load(open(ROOT / 'work/retarget-data/images' / f"{A['id']}.pkl", 'rb'))
        try:
            key, src = engine_source(rows[A['id']], images)
        except ValueError as exc:
            log.append({'case': A['id'], 'skipped': str(exc)}); continue
        target = {'width': int(B['W']), 'height': int(B['H'])}
        ref = designer_reference(src, A, B, p['map'])
        ref_recs = as_records(ref, src, A, p['map'])
        tag = f"{n:02d}"
        imgs = [render(src, out / 'png' / f'{tag}-source.png'), render(ref, out / 'png' / f'{tag}-designer.png')]
        entry = {'case': tag, 'id': A['id'], 'target_id': B['id'], 'title': A['title'], 'from': [A['W'], A['H']], 'to': [B['W'], B['H']]}
        Ana = analyze(src)
        methods = a.methods.split(',')
        for name, opts in (('engine_before', {'learned_retarget': False}), ('engine_model', {'learned_retarget': True})):
            if a.skip_engine or name not in methods:
                imgs.append(None); continue
            cache = out / 'png' / f'{tag}-{name}.meta.json'
            # The engine is deterministic; reuse a finished "before" run for the same case.
            if name == 'engine_before' and cache.exists() and json.load(open(cache)).get('id') == A['id']:
                meta = json.load(open(cache)); design = meta.pop('design'); meta.pop('id')
            else:
                t0 = time.time()
                top = generate(src, target, top_k=1, options=opts)[0]
                design = top['design']
                meta = {'template': top['template'], 'seconds': round(time.time() - t0, 1), 'valid': top['valid']}
                json.dump(dict(meta, design=design, id=A['id']), open(cache, 'w'))
            entry[name] = meta
            imgs.append(render(design, out / 'png' / f'{tag}-{name}.png'))
            m = metrics(as_records(design, src, A, p['map']), ref_recs, B['W'], B['H'])
            if m:
                for k, v in m.items():
                    scores[name][k].append(v)
                entry[name]['metrics'] = {k: round(float(v), 3) for k, v in m.items()}
        g = propose(src, target, Ana, k=1) if 'model_only' in methods else []
        if g:
            imgs.append(render(g[0], out / 'png' / f'{tag}-model.png'))
            m = metrics(as_records(g[0], src, A, p['map']), ref_recs, B['W'], B['H'])
            if m:
                for k, v in m.items():
                    scores['model_only'][k].append(v)
        else:
            imgs.append(None)
        sheet_rows.append((f"{tag} {A['title'][:50]}  {int(A['W'])}x{int(A['H'])} -> {int(B['W'])}x{int(B['H'])}", imgs))
        log.append(entry); print(json.dumps(entry), flush=True)
    for i in range(0, len(sheet_rows), 10):
        sheet(sheet_rows[i:i + 10], labels, out / f'sheet-{i // 10 + 1}.jpg')
    summary = {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} | {'n': len(d['iou'])} for k, d in scores.items()}
    (out / 'results.json').write_text(json.dumps({'summary': summary, 'cases': log}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
