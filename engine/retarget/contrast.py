"""Contrast-aware placement for extreme strips.

Squeezing a design into a strip often slides copy off the panel or photo area it was legible on,
for example white text that ends up over a white region. The engine rejects those layouts as
contrast regressions. This pass finds each regressed text, then moves it to the nearest position
along the strip where it reads at least as the engine requires, using the same cheap composite
estimate as the engine's own diagnostics. A move is kept only if the text lands on the canvas
background or on a layer that fully backs it, and it adds no new block collision. Colours,
sizes and fonts never change.
"""
import copy

import numpy as np

from ..canonical import rotated_extent


def enabled(options=None):
    """Strip legibility fixes (backdrop as cover, copy moved to legible spots): on unless
    options['strip_contrast'] is false or CHAMELEON_STRIP_CONTRAST=0."""
    import os
    o = (options or {}).get('strip_contrast')
    return bool(o) if o is not None else os.environ.get('CHAMELEON_STRIP_CONTRAST', '1') != '0'


def _ok(v, src):
    # the engine's contrast regression is v < 3 and v < 0.6 * source
    return v >= 3 or v >= .6 * src


def _hits(a, b, pad=0.0):
    return a.x < b.x1 + pad and b.x < a.x1 + pad and a.y < b.y1 + pad and b.y < a.y1 + pad


def keep_contrast(source, design, A, steps=48, src=None):
    """design with regressed texts moved to legible positions (a new dict), or design unchanged.
    src: the source's text contrast ratios (quality.text_contrast), when already computed."""
    from ..quality import collisions, composite, painted_prefix, text_contrast
    ratios, _ = text_contrast(design)
    src = src if src is not None else text_contrast(source)[0]
    low = [i for i, v in ratios.items() if not _ok(v, src.get(i, 21))]
    if not low:
        return design
    out = copy.deepcopy(design)
    W, H = float(out['canvas']['width']), float(out['canvas']['height'])
    wide = W >= H
    before = len(collisions(source, design, A))
    covers = {e['id'] for e in out['elements'] if e['width'] >= .95 * W and e['height'] >= .95 * H}
    moved = False
    for tid in low:
        by = {e['id']: e for e in out['elements']}
        t = by[tid]
        if t.get('rotation'):
            continue
        bg, _ = composite(painted_prefix(out, tid))
        others = [e for e in out['elements'] if e['id'] != tid and e['id'] not in covers]
        x0, y0, w, h = t['x'], t['y'], t['width'], t['height']
        m = min(W, H) * 0.04
        if wide:
            along = np.linspace(0, max(0.0, W - w), steps)
            across = sorted({y0, (H - h) / 2, min(max(m, 0), max(0.0, H - h)), max(0.0, H - h - m)})
            spots = [(x, y) for x in along for y in across]
        else:
            along = np.linspace(0, max(0.0, H - h), steps)
            across = sorted({x0, (W - w) / 2, min(m, max(0.0, W - w)), max(0.0, W - w - m)})
            spots = [(x, y) for y in along for x in across]
        spots.sort(key=lambda p: abs(p[0] - x0) + abs(p[1] - y0))
        for x, y in spots:
            if abs(x - x0) < 1 and abs(y - y0) < 1:
                continue
            probe = dict(t, x=float(x), y=float(y))
            box = rotated_extent(probe)
            # land only on the background or on a layer that fully backs the text
            clash = False
            for o in others:
                ob = rotated_extent(o)
                if not _hits(box, ob, 2.0):
                    continue
                backs = o['type'] != 'text' and ob.x <= box.x and ob.y <= box.y and ob.x1 >= box.x1 and ob.y1 >= box.y1
                if not backs:
                    clash = True
                    break
            if clash:
                continue
            v = text_contrast({'canvas': out['canvas'], 'elements': [probe]}, {tid: bg})[0][tid]
            if not _ok(v, src.get(tid, 21)):
                continue
            trial = copy.deepcopy(out)
            next(e for e in trial['elements'] if e['id'] == tid).update(x=float(x), y=float(y))
            if len(collisions(source, trial, A)) > before:
                continue
            out, moved = trial, True
            break
    return out if moved else design
