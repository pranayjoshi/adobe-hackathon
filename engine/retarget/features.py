"""Shared featurization and output binning for the learned retargeting model.

Numpy only: used by training (torch) and by the engine's numpy inference.
An element record is a dict with keys: type (0..4, Crello element types), x, y, w, h,
angle, opacity, text, font_size, lines, align (0..3), caps, line_height.
"""
import math

import numpy as np

N_TYPES = 5
TEXT, IMAGE, BACKGROUND, SVG = 1, 2, 3, 0
MAX_ELEMENTS = 48
FIELDS = ('w', 'h', 'x', 'y', 'fs')
N_BINS = 256
TASK_PAIR, TASK_NOISY, TASK_ARRANGE = 0, 1, 2
N_FEATURES = 32
N_GLOBAL = 6

# Targets are residuals from the per-axis stretch baseline (see base_box): log size ratios and
# centre offsets as a fraction of the target canvas. Bin 128 of each field is "keep the stretch".
LOG_RANGE = (-2.5, 2.5)
OFF_RANGE = (-0.8, 0.8)


def _clip(v, lo, hi):
    return min(hi, max(lo, v))


def is_cover(e, W, H):
    return e['type'] == BACKGROUND or (e['w'] >= 0.95 * W and e['h'] >= 0.95 * H)


def base_box(e, W, H, tW, tH):
    """Per-axis stretch of centres, uniform (geometric-mean) scale of size; covers fill the canvas."""
    sx, sy = tW / W, tH / H
    if is_cover(e, W, H):
        return {'x': e['x'] * sx, 'y': e['y'] * sy, 'w': e['w'] * sx, 'h': e['h'] * sy,
                'font_size': (e.get('font_size') or 0) * math.sqrt(sx * sy)}
    s = math.sqrt(sx * sy)
    w, h = e['w'] * s, e['h'] * s
    k = min(1.0, tW / w, tH / h)
    w, h = w * k, h * k
    cx, cy = (e['x'] + e['w'] / 2) * sx, (e['y'] + e['h'] / 2) * sy
    return {'x': cx - w / 2, 'y': cy - h / 2, 'w': w, 'h': h, 'font_size': (e.get('font_size') or 0) * s * k}


def centered(els, W, H, tW, tH):
    """Arrange-task source: each element at its stretched size, stacked at the target canvas centre."""
    out = []
    for e in els:
        b = base_box(e, W, H, tW, tH)
        if not is_cover(e, W, H):
            b['x'], b['y'] = (tW - b['w']) / 2, (tH - b['h']) / 2
        out.append(dict(e, x=b['x'], y=b['y'], w=b['w'], h=b['h'], font_size=b['font_size']))
    return out


def element_features(e, W, H, rank, n, max_fs, tW, tH):
    x, y, w, h = e['x'] / W, e['y'] / H, e['w'] / W, e['h'] / H
    b = base_box(e, W, H, tW, tH)
    is_text = e['type'] == TEXT
    fs = e.get('font_size') or 0.0
    text = e.get('text') or ''
    lines = max(1, int(e.get('lines') or 1)) if is_text else 0
    chars = len(text)
    longest = max((len(s) for s in text.split('\n')), default=0) if is_text else 0
    align = [0.0] * 4
    if is_text:
        align[int(e.get('align') or 0) % 4] = 1.0
    ang = math.radians(e.get('angle') or 0.0)
    f = [
        x, y, w, h, x + w / 2, y + h / 2,
        _clip(math.log(max(e['w'], 1e-3) / max(e['h'], 1e-3)), -4, 4) / 2,
        _clip(math.log(max(w * h, 1e-6)), -12, 1) / 6,
        float(is_text),
        fs / H * 10 if is_text else 0.0,
        fs / math.sqrt(W * H) * 10 if is_text else 0.0,
        (fs / max_fs) if (is_text and max_fs > 0) else 0.0,
        math.log1p(chars) / 5,
        lines / 10,
        math.log1p(longest) / 4,
        *align,
        float(bool(e.get('caps'))),
        (e.get('line_height') or 0.0) / 2,
        e.get('opacity', 1.0) if e.get('opacity') is not None else 1.0,
        math.sin(ang), math.cos(ang) - 1.0,
        rank / max(1, n - 1),
        float(w > 0.95 and h > 0.95),
        float(x <= 0.01), float(y <= 0.01), float(x + w >= 0.99), float(y + h >= 0.99),
        math.sqrt(b['w'] / tW), math.sqrt(b['h'] / tH),
    ]
    assert len(f) == N_FEATURES
    return f


def encode_source(els, W, H, tW, tH, task=TASK_PAIR):
    """Return (types[n], feats[n, F], glob[4]) for up to MAX_ELEMENTS elements in stacking order."""
    els = els[:MAX_ELEMENTS]
    n = len(els)
    max_fs = max([e.get('font_size') or 0.0 for e in els if e['type'] == TEXT] or [0.0])
    types = np.array([int(e['type']) for e in els], dtype=np.int64)
    feats = np.array([element_features(e, W, H, r, n, max_fs, tW, tH) for r, e in enumerate(els)], dtype=np.float32)
    glob = np.array([math.log(W / H), math.log(tW / tH), math.log(n) / 4] + [float(task == t) for t in range(3)], dtype=np.float32)
    return types, feats, glob


def _to_bin(v, lo, hi):
    return int(round(_clip((v - lo) / (hi - lo), 0.0, 1.0) * (N_BINS - 1)))


def _from_bin(b, lo, hi):
    return lo + (hi - lo) * b / (N_BINS - 1)


def encode_target(t, src, W, H, tW, tH):
    """Bins for (w, h, x, y, fs) of target element t, relative to the stretch of source element src."""
    b = base_box(src, W, H, tW, tH)
    lw = math.log(max(t['w'], 1e-3) / max(b['w'], 1e-3))
    lh = math.log(max(t['h'], 1e-3) / max(b['h'], 1e-3))
    dx = (t['x'] + t['w'] / 2 - b['x'] - b['w'] / 2) / tW
    dy = (t['y'] + t['h'] / 2 - b['y'] - b['h'] / 2) / tH
    fs = 0
    if t['type'] == TEXT and (t.get('font_size') or 0) > 0 and (b['font_size'] or 0) > 0:
        fs = _to_bin(math.log(t['font_size'] / b['font_size']), *LOG_RANGE)
    return [_to_bin(lw, *LOG_RANGE), _to_bin(lh, *LOG_RANGE), _to_bin(dx, *OFF_RANGE), _to_bin(dy, *OFF_RANGE), fs]


def decode_target(bins, src, W, H, tW, tH):
    b = base_box(src, W, H, tW, tH)
    w = b['w'] * math.exp(_from_bin(bins[0], *LOG_RANGE))
    h = b['h'] * math.exp(_from_bin(bins[1], *LOG_RANGE))
    cx = b['x'] + b['w'] / 2 + _from_bin(bins[2], *OFF_RANGE) * tW
    cy = b['y'] + b['h'] / 2 + _from_bin(bins[3], *OFF_RANGE) * tH
    fs = (b['font_size'] or 0) * math.exp(_from_bin(bins[4], *LOG_RANGE))
    return {'x': cx - w / 2, 'y': cy - h / 2, 'w': max(w, 1.0), 'h': max(h, 1.0), 'font_size': fs}
