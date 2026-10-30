"""What a layout looks like once drawn: how much canvas it actually uses.

JSON boxes cannot see this: a box full of transparent image counts as used canvas, and white
on white as content. These are measured on the official renderer's own output.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
NAMES = ('blank', 'ink_span', 'gap_x', 'gap_y', 'off_centre')
NEUTRAL = [0.5, 0.8, 0.2, 0.2, 0.1]   # used if a render fails, so it neither helps nor hurts


def measure(rgb: np.ndarray) -> list[float]:
    """Blank share, narrower ink extent, largest empty column / row run, ink centroid offset."""
    H, W = rgb.shape[:2]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(int)
    q = (border // 16)
    values, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], axis=0, return_counts=True)
    v = int(values[counts.argmax()])
    bg = np.array([v // 256, (v // 16) % 16, v % 16]) * 16 + 8
    ink = (np.abs(rgb.astype(int) - bg).max(axis=2) > 28)
    cols, rows = ink.mean(axis=0) > 0.01, ink.mean(axis=1) > 0.01

    def gap(a):
        best = run = 0
        for x in a:
            run = 0 if x else run + 1
            best = max(best, run)
        return best / max(len(a), 1)

    def extent(a):
        i = np.flatnonzero(a)
        return (i[-1] - i[0] + 1) / len(a) if len(i) else 0.0

    total = ink.sum()
    if total:
        ys, xs = np.nonzero(ink)
        off = max(abs(xs.mean() / W - 0.5), abs(ys.mean() / H - 0.5))
    else:
        off = 0.5
    return [float(1 - ink.mean()), float(min(extent(cols), extent(rows))), gap(cols), gap(rows), float(off)]


def render(design: dict[str, Any], png: Path, max_size: int = 360) -> None:
    from renderer.converter.convert import render_design_to_png
    render_design_to_png(design, REPO / 'brand_kit/assets', png, max_size=max_size)


def features(design: dict[str, Any]) -> list[float]:
    try:
        with tempfile.TemporaryDirectory() as folder:
            png = Path(folder) / 'layout.png'
            render(design, png)
            return measure(np.asarray(Image.open(png).convert('RGB')))
    except Exception:
        return list(NEUTRAL)
