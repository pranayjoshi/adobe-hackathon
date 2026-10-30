"""The judge: a linear model of the blind panel's preferences (tools/final_fit.py fits it).

Features: merged.prefs (type sizes, canvas use, our checks), contrast losses, and final.pixels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from merged import prefs as base
from final import pixels

WEIGHTS = Path(__file__).with_name('weights.json')
NAMES = (*base.NAMES, 'contrast', *pixels.NAMES)


def layout_features(source: dict[str, Any], report: dict[str, Any]) -> list[float]:
    design = report['design']
    contrast = len(report['quality']['defects'].get('contrast_regressions', []))
    return [*base.features(source, design, report), math.log1p(contrast)]


_model = None


def score(features: list[float]) -> float:
    global _model
    if _model is None:
        _model = json.loads(WEIGHTS.read_text())
    return sum(w * (f - m) / s for w, f, m, s in zip(_model['weights'], features, _model['mean'], _model['scale']))
