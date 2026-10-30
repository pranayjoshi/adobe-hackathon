"""Fit final/weights.json on the blind panel's round 1 votes.

    .venv/bin/python -m tools.final_fit [--study work/blind-eval/archive/study-v3-gaurav-v10]

Every option the judges saw is re-drawn at judging size for its pixel features. A vote for a
picture says it beat each other picture shown; "none" and "all bad" votes say nothing about
order. Logistic regression on feature differences, cross-validated grouped by design.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENGINES = ('pranay', 'gaurav', 'merged')
WORK = ROOT / 'work/blind-eval'


def layout(job):
    """One option's full feature vector."""
    from engine.pipeline import analyze, assess_design
    from final import pixels, prefs
    sample, target, path = job
    rec = json.loads(Path(path).read_text())
    if not rec.get('ok'):
        return None
    source = json.loads((WORK / 'crello/designs' / f'{sample}.json').read_text())['source']
    report = assess_design(source, target, rec['design'], analyze(source), 'option')
    return prefs.layout_features(source, report) + pixels.features(rec['design'])


def fit(D, lam=1.0, steps=4000, lr=0.1):
    X = np.vstack([D, -D])
    y = np.r_[np.ones(len(D)), np.zeros(len(D))]
    w = np.zeros(D.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y) / len(y) + lam * w / len(y))
    return w


def cross_validate(Z, groups, folds=5, seed=0):
    designs = sorted(set(groups))
    np.random.default_rng(seed).shuffle(designs)
    fold = {d: i % folds for i, d in enumerate(designs)}
    which = np.array([fold[g] for g in groups])
    good = sum(int(((Z[which == k] @ fit(Z[which != k])) > 0).sum()) for k in range(folds))
    return good / len(Z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, default=WORK / 'archive/study-v3-gaurav-v10')
    a = ap.parse_args()
    from final.prefs import NAMES, WEIGHTS
    pairs = {p['id']: p for p in json.loads((a.study / 'pairs.json').read_text())}
    runs = {'pranay': WORK / 'runs/pranay', 'gaurav': a.study / 'runs-gaurav-ed11607', 'merged': a.study / 'runs-merged'}
    votes = [v for v in sqlite3.connect(a.study / 'votes.sqlite').execute('select pair, winner from votes')
             if v[1] not in ('none', 'all-bad') and v[0] in pairs]
    keys = sorted({(t, e) for t, _ in votes for e in ENGINES})
    jobs = [(pairs[t]['sample'], pairs[t]['target_canvas'], str(runs[e] / f'{t}.json')) for t, e in keys]
    with ProcessPoolExecutor(6) as pool:
        table = dict(zip(keys, pool.map(layout, jobs, chunksize=4)))
    D, groups = [], []
    for task, winner in votes:
        won = set(winner.split(','))
        for w in won:
            for l in ENGINES:
                if l not in won and table[task, l] is not None and table[task, w] is not None:
                    D.append(np.subtract(table[task, w], table[task, l]))
                    groups.append(pairs[task]['sample'])
    D, groups = np.array(D), np.array(groups)
    F = np.array([f for f in table.values() if f is not None])
    mean, scale = F.mean(0), F.std(0) + 1e-6
    Z = D / scale
    accuracy = float(np.mean([cross_validate(Z, groups, seed=s) for s in range(3)]))
    base = float(np.mean([cross_validate(Z[:, :15], groups, seed=s) for s in range(3)]))
    w = fit(Z)
    print(f'{len(D)} preferences, {len(set(groups))} designs; held-out accuracy {accuracy:.3f} (without pixels/contrast {base:.3f})')
    for n, wi in sorted(zip(NAMES, w), key=lambda t: -abs(t[1])):
        print(f'  {n:14} {wi:+.3f}')
    WEIGHTS.write_text(json.dumps({'names': NAMES, 'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': w.tolist(),
                                   'held_out_accuracy': round(accuracy, 3), 'preferences': len(D), 'study': a.study.name}, indent=1))


if __name__ == '__main__':
    main()
