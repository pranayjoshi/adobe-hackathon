"""Learned role classifier: gradient-boosted trees evaluated with numpy only.

    from engine.role_model import predict
    predict(source) -> {element id: (role, confidence)}

The model (`engine/models/roles.npz`, ~1 MB) is a scikit-learn
HistGradientBoostingClassifier trained by tools/train_role_model.py on Crello
designs labelled through DesignAsCode themes (see docs/ROLE_MODEL.md). It is
exported as *complete* binary trees of a fixed depth so that every tree of
every class is traversed at once for every element with a handful of numpy
gathers: a design costs well under 20 ms and the result is bit-for-bit
deterministic (no threads, no randomness).

Inputs are the id-free features of `engine.role_features`. Nothing here reads
element ids or touches the analyzer. When numpy or the model file is missing
`predict` returns {} and callers fall back to the rule-based roles.

Export layout (all arrays indexed [tree] with trees ordered iteration-major,
class-minor; a padded leaf keeps its value in every descendant slot, so the
traversal always runs to depth D):
    feature   int16  [n_trees, 2**D - 1]   feature index at the node (0 for pads)
    threshold float32[n_trees, 2**D - 1]   go left when x[feature] <= threshold
                                           (+inf on pads: always left)
    value     float32[n_trees, 2**D]       leaf value at depth D
    baseline  float32[n_classes]           raw score before any tree
    roles     str    [n_classes]
    feature_names str[n_features]          must equal role_features.FEATURE_NAMES
    depth     int                          D
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from engine import role_features as RF

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "roles.npz"
TIER = {"headline": 0.0, "subhead": 1.0, "cta": 2.0, "badge": 3.0, "body": 3.0, "fine_print": 4.0}

_CACHE: dict[str, Any] = {}


class RoleModel:
    """A loaded model. Use `load()` / `predict()`; this class is also handy in
    tests with a tiny fixture file."""

    def __init__(self, path: Path | str):
        import numpy as np

        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.feature = np.ascontiguousarray(z["feature"].astype(np.int64))
            self.threshold = np.ascontiguousarray(z["threshold"].astype(np.float64))
            self.value = np.ascontiguousarray(z["value"].astype(np.float64))
            self.baseline = z["baseline"].astype(np.float64)
            self.roles = [str(r) for r in z["roles"]]
            self.feature_names = [str(f) for f in z["feature_names"]]
            self.depth = int(z["depth"])
            self.n_iter = int(z["n_iter"]) if "n_iter" in z else self.feature.shape[0] // len(self.roles)
        if self.feature_names != RF.FEATURE_NAMES:
            raise ValueError(f"{self.path.name}: feature set differs from engine.role_features "
                             f"({len(self.feature_names)} vs {len(RF.FEATURE_NAMES)} names)")
        self.n_trees = self.feature.shape[0]
        self.n_classes = len(self.roles)
        if self.n_trees != self.n_iter * self.n_classes:
            raise ValueError(f"{self.path.name}: {self.n_trees} trees for {self.n_iter} x {self.n_classes}")
        self._tree_idx = np.arange(self.n_trees)

    # -- inference -----------------------------------------------------------

    def raw_scores(self, X: Any) -> Any:
        """[n_rows, n_classes] raw (pre-softmax) scores."""
        import numpy as np

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        node = np.zeros((n, self.n_trees), dtype=np.int64)
        tree = np.broadcast_to(self._tree_idx, (n, self.n_trees))
        for _ in range(self.depth):
            f = self.feature[tree, node]                 # [n, trees]
            t = self.threshold[tree, node]
            x = X[np.arange(n)[:, None], f]
            node = 2 * node + 1 + (x > t)
        leaf = node - (2 ** self.depth - 1)
        contrib = self.value[tree, leaf]                  # [n, trees]
        contrib = contrib.reshape(n, self.n_iter, self.n_classes).sum(axis=1)
        return contrib + self.baseline[None, :]

    def proba(self, X: Any) -> Any:
        import numpy as np

        s = self.raw_scores(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def predict_design(self, source: dict[str, Any]) -> dict[str, dict[str, float]]:
        """{id: {role: probability}} for every element of the design."""
        ids, rows = RF.features(source)
        if not ids:
            return {}
        P = self.proba(rows)
        out: dict[str, dict[str, float]] = {}
        for i, eid in enumerate(ids):
            out[eid] = {r: float(P[i, k]) for k, r in enumerate(self.roles)}
        return out


# ---------------------------------------------------------------------------


def load(path: Path | str | None = None) -> RoleModel | None:
    """The cached model, or None when numpy / the file is unavailable."""
    key = str(path or MODEL_PATH)
    if key in _CACHE:
        return _CACHE[key]
    model: RoleModel | None = None
    try:
        if Path(key).exists():
            model = RoleModel(key)
        else:
            logger.info("role model %s not found", key)
    except Exception as exc:  # noqa: BLE001 - never let the model break resize()
        logger.warning("role model unavailable: %r", exc)
        model = None
    _CACHE[key] = model
    return model


def enabled() -> bool:
    """CHAMELEON_ROLE_MODEL=1 turns the learned roles on (default off)."""
    return os.environ.get("CHAMELEON_ROLE_MODEL", "0").strip() not in ("", "0", "false", "no", "off")


def predict_proba(source: dict[str, Any], path: Path | str | None = None) -> dict[str, dict[str, float]]:
    model = load(path)
    if model is None:
        return {}
    try:
        return model.predict_design(source)
    except Exception as exc:  # noqa: BLE001
        logger.warning("role model failed: %r", exc)
        return {}


def predict(source: dict[str, Any], path: Path | str | None = None) -> dict[str, tuple[str, float]]:
    """{element id: (role, confidence)}; confidence is the winning probability.
    Text elements only get text roles, images image roles, shapes shape roles
    (the classifier is trained on all three but a stray cross-type argmax
    would be meaningless to the layout code)."""
    probs = predict_proba(source, path)
    if not probs:
        return {}
    by_id = {str(el.get("id")): el for el in source.get("elements", [])}
    out: dict[str, tuple[str, float]] = {}
    for eid, p in probs.items():
        allowed = allowed_roles(str(by_id.get(eid, {}).get("type", "")))
        best = max(((r, v) for r, v in p.items() if r in allowed), key=lambda t: t[1], default=None)
        if best is None:
            best = max(p.items(), key=lambda t: t[1])
        out[eid] = (best[0], best[1])
    return out


def allowed_roles(etype: str) -> set[str]:
    if etype == "text":
        return {"headline", "subhead", "body", "cta", "badge", "fine_print"}
    if etype == "image":
        return {"logo", "hero", "image", "background", "decorative", "cta", "badge"}
    return {"shape", "decorative", "background", "cta", "badge"}


def text_hierarchy(source: dict[str, Any], probs: dict[str, dict[str, float]] | None = None,
                   path: Path | str | None = None) -> dict[str, int]:
    """Rank of every text element (0 = most prominent): expected hierarchy tier
    under the model's probabilities, ties broken by font size (larger first)."""
    if probs is None:
        probs = predict_proba(source, path)
    texts = [el for el in source.get("elements", []) if el.get("type") == "text"]
    if not texts:
        return {}

    def key(el: dict[str, Any]) -> tuple[float, float, str]:
        p = probs.get(str(el.get("id")), {})
        tier = sum(TIER.get(r, 3.0) * v for r, v in p.items()) if p else 3.0
        return (tier, -float(el.get("font_size", 0.0) or 0.0), str(el.get("id")))

    ranked = sorted(texts, key=key)
    return {str(el["id"]): i for i, el in enumerate(ranked)}


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)
