"""Learned appeal score in [0, 1]: P(designer-quality) for a resize of `src` (see train.py)."""
from pathlib import Path

_M = None


def _load():
    global _M
    if _M is None:
        import joblib
        _M = joblib.load(Path(__file__).resolve().parent / 'appeal_features.joblib')
    return _M


def score(src, design, A, si=None):
    import numpy as np
    import harness
    from .features import features
    m = _load()
    si = si or harness.source_info(src, A)
    f = features(src, design, A, si)
    x = np.array([[f.get(k, np.nan) if f.get(k) is not None else np.nan for k in m['keys']]], dtype=float)
    return float(m['model'].predict_proba(x)[0, 1])
