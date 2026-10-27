"""Pick the preference-round designs: 120 val pairs (round 1) and 120 test pairs (rounds 0 and 2).

Stratified 30 per aspect-change bucket, at most two pairs per family and one per source design, never a
pair from the gate subset. No round design is used for tuning (the tune subset is train-split).
"""
import json, math, random, sys
from pathlib import Path
sys.path.insert(0, 'work/postproc')
import harness
from training.retarget.data import load

designs, pairs = load('work/retarget-data')
sub = harness.subsets()
used_pairs = set(sub['ho'])
used_src = {pairs[p]['src'] for p in sub['ho']} | {i for i, _ in sub['panel']}


def bucket(p):
    A, B = designs[p['src']], designs[p['tgt']]
    la = abs(math.log((A['W'] / A['H']) / (B['W'] / B['H'])))
    return 0 if la < 0.3 else 1 if la < 0.7 else 2 if la < 1.1 else 3


out = {}
for split, seed in (('val', 11), ('test', 12)):
    rng = random.Random(seed)
    cand = [i for i, p in enumerate(pairs) if designs[p['src']]['split'] == split and i not in used_pairs
            and len(designs[p['src']]['els']) <= 48]
    rng.shuffle(cand)
    pick, fams, srcs = [], {}, set()
    per = {b: 0 for b in range(4)}
    for i in cand:
        p = pairs[i]
        b = bucket(p)
        if per[b] >= 30 or fams.get(p['family'], 0) >= 2 or p['src'] in srcs:
            continue
        pick.append(i); fams[p['family']] = fams.get(p['family'], 0) + 1; srcs.add(p['src']); per[b] += 1
    out[split] = pick
    print(split, len(pick), per)
Path('work/postproc/prefs/rounds.json').write_text(json.dumps(out))
