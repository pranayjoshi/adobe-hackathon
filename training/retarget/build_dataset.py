"""Turn fetched Crello layout columns into a retargeting dataset.

Outputs (in --out):
  designs.pkl   list of compact designs (canvas, per-element geometry and text attributes)
  pairs.pkl     designer resize pairs: (src index, tgt index, src->tgt element mapping)
  split.json    family-level split; every design in a held-out family is held out entirely

A "family" is a set of designs whose normalized text contents are identical but whose
canvas aspect ratios differ: the same template resized by its designer to another format.
"""
import argparse
import collections
import hashlib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

TYPES = ['SvgElement', 'TextElement', 'ImageElement', 'ColoredBackground', 'SvgMaskElement']
TEXT = 1
ALIGN = ['', 'left', 'center', 'right']


def norm_text(s):
    return re.sub(r'\s+', ' ', s or '').strip().lower()


def compact(row):
    n = int(row['length'])
    if n == 0:
        return None
    W, H = float(row['canvas_width']), float(row['canvas_height'])
    if W <= 0 or H <= 0:
        return None
    els = []
    for i in range(n):
        t = int(row['type'][i])
        text = row['text'][i] or ''
        if t == TEXT and not text.strip():
            continue
        lines = row['text_line'][i] or []
        els.append({
            'i': i, 'type': t,
            'x': float(row['left'][i]), 'y': float(row['top'][i]),
            'w': max(float(row['width'][i]), 1e-3), 'h': max(float(row['height'][i]), 1e-3),
            'angle': float(row['angle'][i] or 0.0), 'opacity': float(row['opacity'][i] if row['opacity'][i] is not None else 1.0),
            'text': text if t == TEXT else '',
            'font_size': float(row['font_size'][i] or 0.0) if t == TEXT else 0.0,
            'lines': (max(lines) + 1) if (t == TEXT and lines) else (text.count('\n') + 1 if t == TEXT else 0),
            'align': int(row['text_align'][i] or 0) if t == TEXT else 0,
            'caps': bool(row['capitalize'][i]) if t == TEXT else False,
            'line_height': float(row['line_height'][i] or 1.0) if t == TEXT else 0.0,
        })
    if not els:
        return None
    return {'id': row['id'], 'W': W, 'H': H, 'title': row['title'] or '', 'format': int(row['format']),
            'category': int(row['category']), 'els': els}


def text_key(d):
    return tuple(sorted(norm_text(e['text']) for e in d['els'] if e['type'] == TEXT))


def log_aspect(e):
    return math.log(e['w'] / e['h'])


def match(a, b):
    """Element correspondence a->b. Text by content; graphics by type, stacking rank and shape."""
    mapping = {}
    by_text = collections.defaultdict(list)
    for j, e in enumerate(b['els']):
        if e['type'] == TEXT:
            by_text[norm_text(e['text'])].append(j)
    for i, e in enumerate(a['els']):
        if e['type'] == TEXT and by_text[norm_text(e['text'])]:
            mapping[i] = by_text[norm_text(e['text'])].pop(0)
    na, nb = len(a['els']), len(b['els'])
    for t in set(range(len(TYPES))) - {TEXT}:
        ia = [i for i, e in enumerate(a['els']) if e['type'] == t]
        ib = [j for j, e in enumerate(b['els']) if e['type'] == t]
        if not ia or not ib:
            continue
        cost = np.zeros((len(ia), len(ib)))
        for p, i in enumerate(ia):
            for q, j in enumerate(ib):
                ea, eb = a['els'][i], b['els'][j]
                shape = abs(log_aspect(ea) - log_aspect(eb))
                if t == 3:  # backgrounds are stretched to cover; shape says nothing
                    shape = 0.0
                cost[p, q] = abs(i / na - j / nb) * 2.0 + shape
        r, c = linear_sum_assignment(cost)
        for p, q in zip(r, c):
            ok_shape = t == 3 or cost[p, q] < 0.6
            if ok_shape:
                mapping[ia[p]] = ib[q]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='work/retarget-data/raw')
    ap.add_argument('--out', default='work/retarget-data')
    ap.add_argument('--holdout', type=float, default=0.12, help='fraction of families held out for test')
    ap.add_argument('--val', type=float, default=0.06)
    a = ap.parse_args()
    designs = []
    seen = set()
    for f in sorted(Path(a.raw).glob('*.parquet')):
        for row in pq.read_table(f).to_pylist():
            if row['id'] in seen:
                continue
            seen.add(row['id'])
            d = compact(row)
            if d:
                d['crello_split'] = f.name.split('-')[0]
                designs.append(d)
    print('designs', len(designs))

    groups = collections.defaultdict(list)
    for k, d in enumerate(designs):
        key = text_key(d)
        # Text-free designs cannot be grouped reliably by content; group by title + element types.
        if not key:
            key = ('__title__', norm_text(d['title']), tuple(sorted(e['type'] for e in d['els'])))
        groups[key].append(k)
    family_of = {}
    pairs = []
    for key, members in groups.items():
        fam = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
        for k in members:
            family_of[k] = fam
        if len(members) < 2:
            continue
        for s in members:
            for t in members:
                if s == t:
                    continue
                A, B = designs[s], designs[t]
                if abs(math.log((A['W'] / A['H']) / (B['W'] / B['H']))) < 0.05:
                    continue
                m = match(A, B)
                cover_a = len(m) / len(A['els'])
                cover_b = len(m) / len(B['els'])
                if cover_a < 0.8 or cover_b < 0.8:
                    continue
                pairs.append({'src': s, 'tgt': t, 'map': m, 'family': fam, 'coverage': min(cover_a, cover_b)})
    families = sorted(set(family_of.values()))
    pair_fams = sorted({p['family'] for p in pairs})
    # Split families deterministically by hash. Families with pairs are split at the requested
    # rates so the held-out set has enough designer references to score against.
    def bucket(f):
        return int(hashlib.sha256(('retarget-v1:' + f).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    split = {}
    for f in families:
        u = bucket(f)
        split[f] = 'test' if u < a.holdout else ('val' if u < a.holdout + a.val else 'train')
    for d_i, d in enumerate(designs):
        d['family'] = family_of[d_i]
        d['split'] = split[family_of[d_i]]
    out = Path(a.out)
    with open(out / 'designs.pkl', 'wb') as fh:
        pickle.dump(designs, fh)
    with open(out / 'pairs.pkl', 'wb') as fh:
        pickle.dump(pairs, fh)
    stats = {
        'designs': len(designs), 'families': len(families), 'families_with_pairs': len(pair_fams),
        'pairs': len(pairs),
        'pairs_by_split': collections.Counter(split[p['family']] for p in pairs),
        'pair_families_by_split': collections.Counter(split[f] for f in pair_fams),
        'designs_by_split': collections.Counter(d['split'] for d in designs),
    }
    (out / 'split.json').write_text(json.dumps({'families': split, 'stats': stats}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
