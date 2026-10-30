"""Fit merged/prefs.py's preference model on a judged study's votes.

    .venv/bin/python -m tools.blind_eval.fit_prefs work/blind-eval/archive/study-v3-gaurav-v10

Each blind vote for a picture becomes pairwise preferences: the picked engine's layout beats
each other engine's layout shown for that task (identical pictures are credited together, and
"no clear winner" / "all bad" votes say nothing about order). A logistic model on feature
differences (no intercept, L2) is checked by 5-fold cross-validation grouped by design, then
refitted on everything and written to merged/prefs_weights.json.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'work/blind-eval/engines/pranay'))  # our engine's checks
from engine.pipeline import analyze, assess_design  # noqa: E402
from merged.prefs import NAMES, WEIGHTS, features  # noqa: E402

ENGINES = ('pranay', 'gaurav', 'merged')


def load(study: Path, work: Path):
    pairs = {p['id']: p for p in json.loads((study / 'pairs.json').read_text())}
    runs = {'pranay': work / 'runs/pranay', 'gaurav': study / 'runs-gaurav-ed11607', 'merged': study / 'runs-merged'}
    db = sqlite3.connect(study / 'votes.sqlite')
    cache, X, groups, rows = {}, [], [], []
    for judge, task, winner in db.execute('select judge, pair, winner from votes'):
        if winner in ('none', 'all-bad') or task not in pairs:
            continue
        p = pairs[task]
        won = set(winner.split(','))
        source = json.loads((work / 'crello/designs' / f"{p['sample']}.json").read_text())['source']
        def feats(name):
            if (task, name) not in cache:
                rec = json.loads((runs[name] / f'{task}.json').read_text())
                if not rec.get('ok'):
                    cache[task, name] = None
                else:
                    report = assess_design(source, p['target_canvas'], rec['design'], analyze(source), name)
                    cache[task, name] = features(source, rec['design'], report)
            return cache[task, name]
        for w in won:
            for l in ENGINES:
                if l in won or feats(l) is None or feats(w) is None:
                    continue
                X.append(np.subtract(feats(w), feats(l)))
                groups.append(p['sample'])
        rows.append((task, won))
    return np.array(X), np.array(groups), cache, pairs, rows


def fit(D, lam=1.0, steps=3000, lr=0.1):
    """Logistic regression on differences, all labelled 1 (and mirrored), no intercept."""
    Xs = np.vstack([D, -D]); y = np.r_[np.ones(len(D)), np.zeros(len(D))]
    w = np.zeros(D.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xs @ w))
        w -= lr * (Xs.T @ (p - y) / len(y) + lam * w / len(y))
    return w


def main(study: Path):
    work = ROOT / 'work/blind-eval'
    D, groups, cache, pairs, rows = load(study, work)
    allf = np.array([f for f in cache.values() if f is not None])
    mean, scale = allf.mean(0), allf.std(0) + 1e-6
    Z = D / scale
    designs = sorted(set(groups)); rng = np.random.default_rng(0); rng.shuffle(designs)
    fold = {d: i % 5 for i, d in enumerate(designs)}
    correct = total = 0
    for k in range(5):
        tr = np.array([fold[g] != k for g in groups])
        w = fit(Z[tr])
        correct += int(((Z[~tr] @ w) > 0).sum()); total += int((~tr).sum())
    w = fit(Z)
    print(f'{len(D)} pairwise preferences from {len(set(groups))} designs')
    print(f'cross-validated pairwise accuracy: {correct / total:.3f}')
    for n, wi in sorted(zip(NAMES, w), key=lambda t: -abs(t[1])):
        print(f'  {n:14} {wi:+.3f}')
    WEIGHTS.write_text(json.dumps({'names': NAMES, 'mean': mean.tolist(), 'scale': scale.tolist(),
                                   'weights': w.tolist(), 'cv_pairwise_accuracy': round(correct / total, 3),
                                   'preferences': len(D), 'study': study.name}, indent=1))
    print('wrote', WEIGHTS.relative_to(ROOT))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
