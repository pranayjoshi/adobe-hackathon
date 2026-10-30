"""Make any layout pass the structural rules (RULES.md), changing position, size and font size only.

Layouts from a generator that does not hold these rules (bleeding off the canvas, stretched
images, re-tracked text) are rebuilt on untouched copies of the source elements, so no other
field can differ from the source.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from engine.canonical import MUTABLE_FIELDS, rotated_extent
from engine.scene import background_cover_allowed
from engine.typography import fit_text, measure


def _clip_rect(e, W, H):
    x0, y0 = max(0.0, e['x']), max(0.0, e['y'])
    x1, y1 = min(float(W), e['x'] + e['width']), min(float(H), e['y'] + e['height'])
    return x0, y0, x1, y1


def _inside(e, W, H):
    """Shrink (about the centre) and shift until the rotated footprint is on the canvas."""
    b = rotated_extent(e)
    k = min(1.0, W / max(b.w, 1e-6), H / max(b.h, 1e-6))
    if k < 1.0:
        cx, cy = e['x'] + e['width'] / 2, e['y'] + e['height'] / 2
        e['width'], e['height'] = e['width'] * k, e['height'] * k
        e['x'], e['y'] = cx - e['width'] / 2, cy - e['height'] / 2
        b = rotated_extent(e)
    e['x'] += max(0.0, -b.x) - max(0.0, b.x1 - W)
    e['y'] += max(0.0, -b.y) - max(0.0, b.y1 - H)


def _proportions(e, s):
    """Fit the source's aspect inside the element's box, centred."""
    if min(s['width'], s['height'], e['width'], e['height']) <= 0:
        return
    a = s['width'] / s['height']
    w, h = min(e['width'], e['height'] * a), min(e['height'], e['width'] / a)
    e['x'] += (e['width'] - w) / 2
    e['y'] += (e['height'] - h) / 2
    e['width'], e['height'] = w, h


def _room(e, W, H):
    """The largest box this element's rotation can have on the canvas (width, height)."""
    t = math.radians(float(e.get('rotation') or 0.0))
    c, s = abs(math.cos(t)), abs(math.sin(t))
    if s < 0.2:
        return float(W), float(H)
    if c < 0.2:
        return float(H), float(W)
    side = min(W, H) / (c + s)
    return side, side


def _text(e, wanted, W, H):
    """Keep the wanted size if the text fits; else widen its box before shrinking the type.

    The layout's box may have been measured with tighter tracking than the source's, which is
    restored here, so a box that no longer fits says nothing about how small the type must be.
    """
    size = max(1, int(round(wanted)))
    width, height = max(1.0, e['width']), max(1.0, e['height'])
    fit = measure(e, size, width)
    if not fit.overflow and fit.height <= height + 0.5:
        e['font_size'] = size
        return
    room_w, room_h = _room(e, W, H)
    cx, cy = e['x'] + e['width'] / 2, e['y'] + e['height'] / 2
    for k in range(9):
        w = width + (max(width, room_w) - width) * k / 8
        fit = measure(e, size, w)
        if not fit.overflow and fit.height <= room_h + 0.5:
            break
    else:
        fit = fit_text(e, room_w, room_h, size)
        w = room_w
    e['font_size'] = fit.font_size
    e['width'], e['height'] = min(w, room_w), min(max(fit.height, 1.0), room_h)
    e['x'], e['y'] = cx - e['width'] / 2, cy - e['height'] / 2


def comply(source: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    W, H = design['canvas']['width'], design['canvas']['height']
    laid = {e['id']: e for e in design['elements']}
    out = []
    for s in source['elements']:
        e = copy.deepcopy(s)
        got = laid.get(s['id'])
        if got:
            e.update({k: got[k] for k in MUTABLE_FIELDS if k in got})
        if background_cover_allowed(source, s, {**e, 'fit': 'cover'}) and (got or {}).get('fit') == 'cover':
            e.update(x=0, y=0, width=W, height=H, fit='cover')
            out.append(e)
            continue
        rotated = abs(float(e.get('rotation') or 0.0)) > 1e-6
        if not rotated:
            x0, y0, x1, y1 = _clip_rect(e, W, H)
            if x1 - x0 >= 1 and y1 - y0 >= 1:   # an edge that bled off is trimmed back to the canvas
                e.update(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        if e['type'] == 'image' and s.get('fit') != 'cover':
            _proportions(e, s)
        if e['type'] == 'text':
            _text(e, (got or s)['font_size'], W, H)
        _inside(e, W, H)
        e['width'], e['height'] = max(1.0, e['width']), max(1.0, e['height'])
        _inside(e, W, H)
        out.append(e)
    return {'canvas': {'width': W, 'height': H}, 'elements': out}
