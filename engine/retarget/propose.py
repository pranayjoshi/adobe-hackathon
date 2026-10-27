"""Learned retargeting proposals for the engine.

The model (trained on Crello designer resizes) predicts each element's target box and font
size. Hard rules stay here, in code: graphics keep their aspect ratio, backgrounds cover the
canvas, multi-element units (buttons, explicit groups) move rigidly, text is re-measured with
the renderer's metrics so it never overflows, and copy keeps a readable floor. The proposals
are then de-overlapped in source order (deoverlap.py) and judged like every other candidate.
"""
import copy
import math
from ..blocks import FLOORS
from ..canonical import backdrop_patch, rotated_extent
from ..scene import background_cover_allowed
from ..typography import fit_text, measure
from .deoverlap import deoverlap
from .features import BACKGROUND, IMAGE, MAX_ELEMENTS, SVG, TEXT
from .infer import predict, weights_path

ALIGN = {'left': 1, 'center': 2, 'right': 3, 'justify': 1}


def available():
    return weights_path().exists()


def _covers(el, W, H):
    """Geometrically covers the canvas (the training definition; inset panels are not covers)."""
    return float(el['x']) <= 0.03 * W and float(el['y']) <= 0.03 * H and \
        float(el['x']) + float(el['width']) >= 0.97 * W and float(el['y']) + float(el['height']) >= 0.97 * H


def records(source, A):
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    order = sorted(source['elements'], key=lambda e: (e.get('z_index', 0)))
    out = []
    for el in order:
        cover = _covers(el, W, H)
        r = {'id': el['id'], 'x': float(el['x']), 'y': float(el['y']), 'w': max(float(el['width']), 1e-3),
             'h': max(float(el['height']), 1e-3), 'angle': float(el.get('rotation') or 0.0),
             'opacity': float(el.get('opacity', 1.0) if el.get('opacity') is not None else 1.0), 'cover': cover}
        if el['type'] == 'text':
            fs = float(el.get('font_size') or 12)
            content = str(el.get('content', ''))
            r.update(type=TEXT, text=content, font_size=fs,
                     lines=measure(el, int(fs), float(el['width'])).lines,
                     align=ALIGN.get(el.get('text_align', 'left'), 1),
                     caps=bool(el.get('capitalize')) or (content.isupper() and len(content) > 3),
                     line_height=float(el.get('line_height') or 1.2))
        elif el['type'] == 'image':
            r.update(type=IMAGE)
        else:
            r.update(type=BACKGROUND if cover else SVG)
        out.append(r)
    return out


def _units(source, A):
    """Multi-element units that must move rigidly: analysis blocks and explicit group_ids."""
    units, claimed = [], set()
    for b in list(A.blocks) + list(A.stickers) + list(A.accents):
        ids = [m['id'] for m in b.members]
        if len(ids) > 1 and not claimed & set(ids):
            units.append(ids); claimed |= set(ids)
    groups = {}
    for el in source['elements']:
        if el.get('group_id') and el['id'] not in claimed:
            groups.setdefault(el['group_id'], []).append(el['id'])
    units += [ids for ids in groups.values() if len(ids) > 1]
    return units


def _union(boxes):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _floor(A, el):
    return FLOORS.get(A.roles.get(el['id'], 'body'), 8)


def _contain_rotated(el, x, y, w, h, tW, tH):
    """Keep the rotated footprint on the canvas (the invariant checks the rotated extent)."""
    if not el.get('rotation'):
        return x, y, w, h
    ext = rotated_extent({'x': x, 'y': y, 'width': w, 'height': h, 'rotation': el['rotation']})
    k = min(1.0, (tW - 1) / max(ext.w, 1e-6), (tH - 1) / max(ext.h, 1e-6))
    if k < 1.0 and el['type'] != 'text':
        cx, cy = x + w / 2, y + h / 2
        w, h = w * k, h * k
        x, y = cx - w / 2, cy - h / 2
        ext = rotated_extent({'x': x, 'y': y, 'width': w, 'height': h, 'rotation': el['rotation']})
    x += max(0.0, -ext.x) - max(0.0, ext.x + ext.w - tW)
    y += max(0.0, -ext.y) - max(0.0, ext.y + ext.h - tH)
    return x, y, w, h


def materialize(source, target, A, recs, pred):
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    tW, tH = float(target['width']), float(target['height'])
    by_id = {e['id']: e for e in source['elements']}
    box = {}
    fs = {}
    covers = []
    for r, p in zip(recs, pred):
        el = by_id[r['id']]
        if r['cover']:
            covers.append((r, p)); continue
        cx, cy = p['x'] + p['w'] / 2, p['y'] + p['h'] / 2
        if r['type'] == TEXT:
            box[r['id']] = (p['x'], p['y'], p['w'], p['h'])
            fs[r['id']] = p['font_size']
        else:
            a = r['w'] / r['h']; area = p['w'] * p['h']
            w, h = math.sqrt(area * a), math.sqrt(area / a)
            box[r['id']] = (cx - w / 2, cy - h / 2, w, h)
    # Backgrounds. The rules allow no off-canvas elements and no distortion; the one sanctioned way
    # to fill a canvas with a photo is the schema's cover crop. Otherwise contain (or fill, for shapes).
    crop = set()
    for r, p in covers:
        el = by_id[r['id']]
        if background_cover_allowed(source, el, {**el, 'fit': 'cover'}):
            crop.add(r['id']); box[r['id']] = (0.0, 0.0, tW, tH)
        else:
            b = backdrop_patch(el, tW, tH)
            box[r['id']] = (b['x'], b['y'], b['width'], b['height'])
    # Rigid units: keep the source arrangement, scaled into the predicted union.
    for ids in _units(source, A):
        ids = [i for i in ids if i in box]
        if len(ids) < 2:
            continue
        sb = _union([(float(by_id[i]['x']), float(by_id[i]['y']), float(by_id[i]['width']), float(by_id[i]['height'])) for i in ids])
        pb = _union([box[i] for i in ids])
        s = math.sqrt(max(pb[2] * pb[3], 1e-6) / max(sb[2] * sb[3], 1e-6))
        s = min(s, tW / sb[2], tH / sb[3])
        cx, cy = pb[0] + pb[2] / 2, pb[1] + pb[3] / 2
        for i in ids:
            e = by_id[i]
            box[i] = (cx + (float(e['x']) - sb[0] - sb[2] / 2) * s, cy + (float(e['y']) - sb[1] - sb[3] / 2) * s,
                      float(e['width']) * s, float(e['height']) * s)
            if e['type'] == 'text':
                fs[i] = float(e['font_size']) * s
    elements = []
    for el in source['elements']:
        out = copy.deepcopy(el)
        x, y, w, h = box.get(el['id'], (float(el['x']) * tW / W, float(el['y']) * tH / H, float(el['width']) * tW / W, float(el['height']) * tH / H))
        if el['type'] == 'text':
            w = min(max(w, 1.0), tW)
            size = max(_floor(A, el), round(fs.get(el['id'], float(el['font_size']))))
            size = min(size, int(el['font_size'] * 4) + 1)
            fit = fit_text(el, w, max(h * 1.6, size * 1.3), size, fs_min=min(size, _floor(A, el)))
            if fit.overflow or fit.height > tH:
                fit = fit_text(el, w, tH * 0.9, size, fs_min=1)
            cy = y + h / 2
            h = fit.height
            y = cy - h / 2
            out['font_size'] = int(fit.font_size)
            # Copy stays fully on the canvas.
            x = min(max(x, 0.0), tW - w); y = min(max(y, 0.0), tH - h)
        elif el['id'] not in crop:
            # Nothing may leave the canvas: shrink uniformly if needed, then shift inside.
            k = min(1.0, tW / max(w, 1e-6), tH / max(h, 1e-6))
            if k < 1.0:
                cx, cy = x + w / 2, y + h / 2
                w, h = w * k, h * k
                x, y = cx - w / 2, cy - h / 2
            x = min(max(x, 0.0), tW - w); y = min(max(y, 0.0), tH - h)
        if el['id'] in crop:
            out['fit'] = 'cover'
        else:
            x, y, w, h = _contain_rotated(el, x, y, w, h, tW, tH)
        out.update(x=round(x, 2), y=round(y, 2), width=round(w, 2), height=round(h, 2))
        elements.append(out)
    return {'canvas': {'width': int(tW), 'height': int(tH)}, 'elements': elements}


def propose(source, target, A, k=6, temperature=0.8, seed=0):
    """Return up to k learned candidate designs (sample 0 is the model's greedy decode)."""
    if not available() or len(source['elements']) > MAX_ELEMENTS:
        return []
    recs = records(source, A)
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    preds = predict(recs, W, H, float(target['width']), float(target['height']), k=k, temperature=temperature, seed=seed)
    return [deoverlap(source, materialize(source, target, A, recs, p), A) for p in preds]


def candidates(source, target, A, options, events):
    """(template, name, description, design) proposals, as the pipeline and providers assemble them."""
    import os
    if not options.get('learned_retarget', os.environ.get('CHAMELEON_LEARNED_RETARGET', '1') != '0') \
            or source['canvas'] == target:
        return []
    try:
        return [(f'learned_{i}', 'Learned layout', 'Predicted by the retargeting model trained on designer resizes', d)
                for i, d in enumerate(propose(source, target, A, k=int(options.get('learned_samples', 6))))]
    except (ValueError, OSError, KeyError) as exc:
        events.append({'learned_retarget': f'unavailable: {exc}'})
        return []
