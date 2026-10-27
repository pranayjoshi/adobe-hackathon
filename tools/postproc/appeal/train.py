"""Train the features appeal scorer: python work/postproc/appeal/train.py
Pointwise GBDT (designer resize = 1, flawed version = 0), split by pair. Reports, on held-out pairs,
how often the designer's version outranks each kind of flaw, and how today's pipeline output scores."""
import collections, json, math, random, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / 'examples.jsonl')]
rows = [r for r in rows if 'f' in r]
keys = sorted({k for r in rows for k in r['f']})
# features that only restate the label-generation process are dropped
keys = [k for k in keys if k not in ('aspect_log',)]
X = lambda rs: np.array([[r['f'].get(k, np.nan) if r['f'].get(k) is not None else np.nan for k in keys] for r in rs], dtype=float)
pairs = sorted({r['pair'] for r in rows})
random.Random(0).shuffle(pairs)
val = set(pairs[: len(pairs) // 5])
tr = [r for r in rows if r['pair'] not in val and r['label'] is not None]
va = [r for r in rows if r['pair'] in val]
clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
clf.fit(X(tr), np.array([r['label'] for r in tr]))
s = clf.predict_proba(X(va))[:, 1]
for r, v in zip(va, s):
    r['s'] = float(v)
by = collections.defaultdict(list)
for r in va:
    by[r['pair']].append(r)
acc = collections.defaultdict(lambda: [0, 0])
cur_vs = [0, 0]
for p, rs in by.items():
    good = [r for r in rs if r['kind'] == 'designer']
    if not good:
        continue
    g = good[0]['s']
    for r in rs:
        if r['label'] == 0:
            k = r['kind'].split(':')[1]
            acc[k][0] += g > r['s']; acc[k][1] += 1
            acc['ALL'][0] += g > r['s']; acc['ALL'][1] += 1
        if r['kind'] == 'current':
            cur_vs[0] += g > r['s']; cur_vs[1] += 1
print('held-out pairs', len(by), 'features', len(keys))
for k, (a, n) in sorted(acc.items(), key=lambda x: -x[1][1]):
    print(f'  designer beats {k:12} {a / n:.3f}  (n={n})')
print(f'  designer beats today\'s output {cur_vs[0] / max(cur_vs[1], 1):.3f} (n={cur_vs[1]})')
joblib.dump({'model': clf, 'keys': keys}, HERE / 'appeal_features.joblib')
print('saved')
