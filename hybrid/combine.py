"""Combine two finished layouts: repair each engine's known weakness, then keep the cleaner one.

The blind evaluation (tools/blind_eval) found the two engines fail in opposite ways:

* ours keeps phrases, text-on-panel units and photos intact, but leaves letter spacing at its
  source pixel value when text shrinks, and falls back to shrinking the whole design into
  the middle of the canvas when no re-layout passes its guardrails;
* Gaurav's fills the canvas with bold, role-driven layouts, but stretches photo backgrounds
  (`fit: fill`) and often splits phrases, collides text or lifts text off its panel.

So each candidate is repaired (`fix_spacing`, `fix_backgrounds`), measured for the defects
people notice (`defects`), and the one with the lower defect cost wins, with canvas use as
the tie-break (`choose`). Nothing here imports either engine: it is geometry plus Pillow.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

BACKGROUND_SHARE = 0.85   # a layer covering this much of the source canvas is a background
WEIGHTS = {'text_collision': 3.0, 'off_panel': 3.0, 'split_phrase': 2.0, 'cut': 3.0, 'letterbox': 2.0, 'tiny_text': 1.0, 'sparse': 1.0}


def _box(e):
    return e['x'], e['y'], e['x'] + e['width'], e['y'] + e['height']


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _gap(a, b):
    """Edge-to-edge distance between two boxes (0 when they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _texts(design):
    return [e for e in design['elements'] if e['type'] == 'text' and (e.get('content') or '').strip()]


def fix_spacing(source: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Scale letter spacing with font size, as tracking is defined relative to the type size.

    Spacing only ever shrinks here: the engine measured each text box with the source
    spacing, so widening it after the fact could push a line past its box.
    """
    src = {e['id']: e for e in source['elements']}
    out = copy.deepcopy(design)
    for e in out['elements']:
        s = src.get(e['id'])
        if e['type'] != 'text' or not s or not s.get('letter_spacing') or not s.get('font_size'):
            continue
        scaled = s['letter_spacing'] * e['font_size'] / s['font_size']
        if abs(scaled) < abs(e.get('letter_spacing') or 0):
            e['letter_spacing'] = round(scaled, 3)
    return out


@lru_cache(maxsize=4096)
def stretch_safe(asset: str) -> bool:
    """True when stretching an image cannot be seen: flat colour, soft gradient or a frame.

    Photos and illustrations have detail everywhere and are cropped (`cover`) instead.
    A frame is drawn at the edges with an empty middle, so stretching it keeps it a frame.
    """
    try:
        with Image.open(asset) as im:
            im = im.convert('RGBA').resize((64, 64))
    except OSError:
        return False
    px = im.load()
    centre = [px[x, y] for x in range(16, 48) for y in range(16, 48)]
    if sum(p[3] < 32 for p in centre) / len(centre) > 0.6:
        return True
    grey = [[(px[x, y][0] * 299 + px[x, y][1] * 587 + px[x, y][2] * 114) // 1000 for x in range(64)] for y in range(64)]
    detail = sum(abs(grey[y][x] - grey[y][x + 1]) + abs(grey[y][x] - grey[y + 1][x])
                 for y in range(63) for x in range(63)) / (2 * 63 * 63)
    return detail < 3.0


def fix_backgrounds(source: dict[str, Any], design: dict[str, Any], assets_dir: Path) -> dict[str, Any]:
    """Crop, rather than stretch, any reshaped image whose stretching would show."""
    src = {e['id']: e for e in source['elements']}
    out = copy.deepcopy(design)
    for e in out['elements']:
        s = src.get(e['id'])
        if e['type'] != 'image' or not s or e.get('fit', 'fill') != 'fill' or not e.get('asset_id'):
            continue
        reshaped = abs((e['width'] / e['height']) / (s['width'] / s['height']) - 1) > 0.03
        if reshaped and not stretch_safe(str(Path(assets_dir) / e['asset_id'])):
            e['fit'] = 'cover'
    return out


def defects(source: dict[str, Any], design: dict[str, Any]) -> dict[str, float]:
    """Counts of the problems the blind review showed people notice, each new in the output."""
    sw, sh = source['canvas']['width'], source['canvas']['height']
    tw, th = design['canvas']['width'], design['canvas']['height']
    S = {e['id']: e for e in source['elements']}
    O = {e['id']: e for e in design['elements']}
    canvas_s, canvas_o = (0, 0, sw, sh), (0, 0, tw, th)
    backgrounds = {i for i, e in S.items() if _area(_box(e)) >= BACKGROUND_SHARE * sw * sh}
    texts = [e['id'] for e in _texts(source) if e['id'] in O]
    found = dict.fromkeys(WEIGHTS, 0.0)

    def overlaps(D, a, b):
        A, B = _box(D[a]), _box(D[b])
        return _inter(A, B) / max(1e-6, min(_area(A), _area(B))) > 0.15
    for k, a in enumerate(texts):
        for b in texts[k + 1:]:
            if overlaps(O, a, b) and not overlaps(S, a, b):
                found['text_collision'] += 1

    # Whatever sat on a panel must still sit on it: a text on its plate, pill or badge, or a
    # photo in its frame (only photos that fill a good part of the frame, not specks on it).
    # Against a full-size layer the only defect is straddling its edge.
    held = texts + [i for i in S if S[i]['type'] != 'text' and i not in backgrounds and i in O]
    for t in held:
        tb = _box(S[t])
        for p, pe in S.items():
            if p == t or pe['type'] == 'text' or p not in O or S[p].get('z_index', 0) > S[t].get('z_index', 0):
                continue
            pb = _box(pe)
            if _inter(tb, pb) < 0.9 * _area(tb) or (S[t]['type'] != 'text' and not 0.25 <= _area(tb) / _area(pb) < 0.95):
                continue
            inside = _inter(_box(O[t]), _box(O[p])) / max(1e-6, _area(_box(O[t])))
            if (p not in backgrounds and inside < 0.7) or (p in backgrounds and 0.05 < inside < 0.7):
                found['off_panel'] += 1
                break

    # Texts that read as one phrase (stacked or side by side, close together) stay close.
    short_s, short_o = min(sw, sh), min(tw, th)
    for k, a in enumerate(texts):
        for b in texts[k + 1:]:
            if _gap(_box(S[a]), _box(S[b])) <= 0.03 * short_s and _gap(_box(O[a]), _box(O[b])) > 0.25 * short_o + 0.1 * max(tw, th):
                found['split_phrase'] += 1

    for i, e in S.items():
        if i in backgrounds or i not in O:
            continue
        seen_s = _inter(_box(e), canvas_s) / max(1e-6, _area(_box(e)))
        seen_o = _inter(_box(O[i]), canvas_o) / max(1e-6, _area(_box(O[i])))
        if seen_s >= 0.95 and seen_o < 0.8:
            found['cut'] += 1

    # The design shrunk into the middle: a background that no longer reaches the edges.
    if backgrounds:
        cover = max(_inter(_box(O[i]), canvas_o) for i in backgrounds if i in O) / (tw * th)
        found['letterbox'] = 1.0 if cover < 0.98 else 0.0

    floor = max(8.0, 0.035 * min(tw, th))
    found['tiny_text'] = sum(O[t]['font_size'] < floor and S[t]['font_size'] >= floor for t in texts) / max(1, len(texts)) * 3

    content = [_box(O[i]) for i in S if i not in backgrounds and i in O and _inter(_box(O[i]), canvas_o) > 0]
    if content:
        x0, y0 = max(0, min(b[0] for b in content)), max(0, min(b[1] for b in content))
        x1, y1 = min(tw, max(b[2] for b in content)), min(th, max(b[3] for b in content))
        found['sparse'] = 1.0 if _area((x0, y0, x1, y1)) / (tw * th) < 0.45 else 0.0
    return found


def cost(found: dict[str, float]) -> float:
    return sum(WEIGHTS[k] * min(v, 3) for k, v in found.items())


def presence(source: dict[str, Any], design: dict[str, Any]) -> float:
    """Tie-break: how boldly the layout uses the canvas (headline size and content extent)."""
    tw, th = design['canvas']['width'], design['canvas']['height']
    O = {e['id']: e for e in design['elements']}
    texts = _texts(source)
    head = max(texts, key=lambda e: e['font_size'])['id'] if texts else None
    headline = min(1.0, O[head]['font_size'] / (0.2 * min(tw, th))) if head in O else 0.0
    boxes = [_box(e) for e in design['elements'] if _area(_box(e)) < 0.85 * tw * th]
    extent = 0.0
    if boxes:
        x0, y0 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes))
        x1, y1 = min(tw, max(b[2] for b in boxes)), min(th, max(b[3] for b in boxes))
        extent = _area((x0, y0, x1, y1)) / (tw * th)
    return 0.5 * headline + 0.5 * extent


def choose(source: dict[str, Any], candidates: dict[str, dict[str, Any] | None], assets_dir: Path) -> dict[str, Any]:
    """Repair every candidate, then keep the one with the lowest defect cost.

    `candidates` maps an engine name to its design, or None when that engine gave none.
    Returns the chosen design plus the per-candidate report, so the choice is explainable.
    """
    report = {}
    for name, design in candidates.items():
        if design is None:
            report[name] = {'ok': False}
            continue
        fixed = fix_backgrounds(source, fix_spacing(source, design), assets_dir)
        found = defects(source, fixed)
        report[name] = {'ok': True, 'design': fixed, 'defects': found, 'cost': cost(found), 'presence': presence(source, fixed)}
    usable = [n for n, r in report.items() if r['ok']]
    if not usable:
        raise ValueError('No engine produced a layout')
    # Lower cost wins; within half a point the bolder layout wins.
    best = min(usable, key=lambda n: (round(report[n]['cost'] * 2) / 2, -report[n]['presence']))
    return {'design': report[best]['design'], 'chosen': best,
            'report': {n: {k: v for k, v in r.items() if k != 'design'} for n, r in report.items()}}
