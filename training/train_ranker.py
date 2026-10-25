"""Bradley-Terry candidate ranker with family holdout and sign constraints.

  python -m training.train_ranker output/workspace/labels.jsonl [--synthetic output/training/perturb/pairs.jsonl]

P(A > B) = sigmoid(w . (x_A - x_B)) on standardised RANK_NAMES features.
Weights on NONNEGATIVE features are bounded at >= 0, so the ranker can never
prefer more defects. Synthetic pairs (clean > perturbed) are down-weighted.
Pairs where either side fails rubric Tier 0-1 are dropped. Output goes to
output/calibration/ranker.json; copying it to engine/models/ is a manual step.
"""
import argparse, json
from pathlib import Path
from engine.learning import NONNEGATIVE, RANK_NAMES, rank_features


def _gated(side):
    rub = side.get('rubric')
    return side.get('valid', True) and (rub is None or rub.get('gate', True))


def load(path, kind=None):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if path else []
    return [r for r in rows if r['decision'] in ('a', 'b') and (kind is None or r.get('kind') == kind) and _gated(r['a']) and _gated(r['b'])]


def train(human, synthetic=(), out=Path('output/calibration/ranker.json'), synthetic_weight=.3, l2=1.0, allow_synthetic_only=False, split_manifest=None):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    rows = [dict(r, w=1.0) for r in human] + [dict(r, w=synthetic_weight) for r in synthetic]
    families = sorted({r['family'] for r in rows})
    if len(human) < 20 and not allow_synthetic_only:
        raise ValueError('Need at least 20 decisive human pairs (use --allow-synthetic-only for a non-deployable development fit)')
    if len(families) < 5:
        raise ValueError('Need pairs from at least five source families')
    if split_manifest is not None:
        from training.families import split_of
        manifest=json.loads(Path(split_manifest).read_text()) if isinstance(split_manifest,(str,Path)) else split_manifest
        assignments={family:split_of(family,manifest) for family in families}
        train_rows=[r for r in rows if assignments[r['family']]=='train']
        test=[r for r in rows if assignments[r['family']]=='test']
        hold={family for family in families if assignments[family]!='train'}
    else:
        # Historical development fit remains reproducible, never promoted.
        manifest=None;hold=set(families[::5]);train_rows=[r for r in rows if r['family'] not in hold];test=[r for r in rows if r['family'] in hold]
    if not train_rows:raise ValueError('No training pairs after family isolation')
    feats = np.array([rank_features(r[s]) for r in train_rows for s in ('a', 'b')]); mean = feats.mean(axis=0); scale = feats.std(axis=0); scale[scale < 1e-6] = 1
    def matrix(items):
        X = np.array([(np.array(rank_features(r['a'])) - np.array(rank_features(r['b']))) / scale for r in items]).reshape(-1, len(RANK_NAMES))
        return X, np.array([float(r['decision'] == 'a') for r in items]), np.array([r['w'] for r in items])
    X, y, sw = matrix(train_rows)
    def loss(w):
        z = X @ w; return (sw * (np.logaddexp(0, z) - y * z)).sum() + .5 * l2 * (w @ w)
    def grad(w):
        return X.T @ (sw * (expit(X @ w) - y)) + l2 * w
    bounds = [(0, None) if n in NONNEGATIVE else (None, None) for n in RANK_NAMES]
    fit = minimize(loss, np.zeros(len(RANK_NAMES)), jac=grad, method='L-BFGS-B', bounds=bounds)
    if not fit.success:
        raise ValueError('Fit did not converge: ' + str(fit.message))
    def accuracy(items):
        if not items:
            return None
        tx, ty, _ = matrix(items); return float(((tx @ fit.x > 0) == ty).mean())
    weights = dict(zip(RANK_NAMES, map(float, fit.x)))
    report = {'human_pairs': len(human), 'synthetic_pairs': len(synthetic), 'families': len(families), 'held_out_families': len(hold),
              'held_out_human_accuracy': accuracy([r for r in test if r.get('kind') == 'human']),
              'held_out_synthetic_accuracy': accuracy([r for r in test if r.get('kind') == 'synthetic']),
              'suspicious_weights': [n for n in ('readability', 'composition', 'hierarchy') if weights[n] < 0],
              'deployable': False,
              'split_manifest_supplied':manifest is not None,
              'promotion_status':'requires independent human evaluation and explicit artifact promotion',
              'deployment': 'Deploy only if held-out human accuracy is within 5 points of human-human agreement and the benchmark release gate passes with it enabled.'}
    import hashlib
    split_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest is not None else None
    model = {'version':1,'features': RANK_NAMES, 'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': fit.x.tolist(), 'report': report,'split_manifest_hash':split_hash}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(model, indent=2)); print(json.dumps(report, indent=2))
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('labels', type=Path, nargs='?'); ap.add_argument('--synthetic', type=Path)
    ap.add_argument('--output', type=Path, default=Path('output/calibration/ranker.json')); ap.add_argument('--synthetic-weight', type=float, default=.3)
    ap.add_argument('--allow-synthetic-only', action='store_true');ap.add_argument('--split-manifest',type=Path); a = ap.parse_args()
    train(load(a.labels, 'human') if a.labels and a.labels.exists() else [], load(a.synthetic, 'synthetic') if a.synthetic else [], a.output,
          a.synthetic_weight, allow_synthetic_only=a.allow_synthetic_only,split_manifest=a.split_manifest)
