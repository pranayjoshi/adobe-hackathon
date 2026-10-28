"""Learned retargeting proposals for the engine.

The model (trained on Crello designer resizes) predicts each element's target box and font
size. Hard rules stay here, in code: graphics keep their aspect ratio, backgrounds cover the
canvas, multi-element units (buttons, explicit groups) move rigidly, text is re-measured with
the renderer's metrics so it never overflows, and copy keeps a readable floor. The proposals
are then de-overlapped in source order (deoverlap.py) and judged like every other candidate.
"""
import copy
import math

import numpy as np

from ..blocks import FLOORS
from ..canonical import backdrop_patch, rotated_extent
from ..scene import background_cover_allowed
from ..typography import fit_text, measure
from .deoverlap import backed, deoverlap, margin, merge
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


def _units(source, A, loose=False):
    """Multi-element units that must move rigidly: analysis blocks, explicit group_ids, and
    backings with the copy and small graphics that sit on them."""
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
    return [u for u in merge(units + backed(source, loose)) if len(u) > 1]


def _union(boxes):
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _clamp(v, lo, hi, fallback):
    """v clamped to [lo, hi]; fallback when the range is empty."""
    return fallback if lo > hi else min(max(v, lo), hi)


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


def materialize(source, target, A, recs, pred, loose=False):
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    tW, tH = float(target['width']), float(target['height'])
    by_id = {e['id']: e for e in source['elements']}
    box = {}
    fs = {}
    anchor = {}
    edge = {}  # copy in a unit uses the unit's margin
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
    for ids in _units(source, A, loose):
        ids = [i for i in ids if i in box]
        if len(ids) < 2:
            continue
        sb = _union([(float(by_id[i]['x']), float(by_id[i]['y']), float(by_id[i]['width']), float(by_id[i]['height'])) for i in ids])
        pb = _union([box[i] for i in ids])
        # Scale: the predicted font sizes when the unit holds copy (they track the designer far
        # better than predicted graphic areas), else the predicted area.
        texts = [by_id[i] for i in ids if by_id[i]['type'] == 'text']
        if texts:
            s = float(np.median([fs[e['id']] / max(float(e['font_size']), 1.0) for e in texts]))
        else:
            s = math.sqrt(max(pb[2] * pb[3], 1e-6) / max(sb[2] * sb[3], 1e-6))
        # Copy in the unit stays readable: grow the whole unit rather than shrink its text.
        s = max([s] + [_floor(A, e) / max(float(e['font_size']), 1.0) for e in texts])
        s = min(s, tW / sb[2], tH / sb[3])
        cx, cy = pb[0] + pb[2] / 2, pb[1] + pb[3] / 2
        # The whole unit goes on the canvas, copy inside its safe margin (per-element clamping
        # later would pull it apart).
        m = margin(sb, W, H, tW, tH) if texts else 0.0
        cx = _clamp(cx, m + sb[2] * s / 2, tW - m - sb[2] * s / 2, tW / 2)
        cy = _clamp(cy, m + sb[3] * s / 2, tH - m - sb[3] * s / 2, tH / 2)
        for i in ids:
            e = by_id[i]
            box[i] = (cx + (float(e['x']) - sb[0] - sb[2] / 2) * s, cy + (float(e['y']) - sb[1] - sb[3] / 2) * s,
                      float(e['width']) * s, float(e['height']) * s)
            if e['type'] == 'text':
                fs[i] = float(e['font_size']) * s
                edge[i] = m
                # Re-measured copy grows away from the unit's centre, so lines stacked against
                # each other (or against their badge's edge) keep the source's spacing.
                anchor[i] = 'bottom' if float(e['y']) + float(e['height']) / 2 < sb[1] + sb[3] / 2 else 'top'
    elements = []
    for el in source['elements']:
        out = copy.deepcopy(el)
        x, y, w, h = box.get(el['id'], (float(el['x']) * tW / W, float(el['y']) * tH / H, float(el['width']) * tW / W, float(el['height']) * tH / H))
        if el['type'] == 'text':
            m = edge.get(el['id'], margin((float(el['x']), float(el['y']), float(el['width']), float(el['height'])), W, H, tW, tH))
            w = min(max(w, 1.0), tW - 2 * m)
            size = max(_floor(A, el), round(fs.get(el['id'], float(el['font_size']))))
            size = min(size, int(el['font_size'] * 4) + 1)
            # Height budget: the predicted box, or the source's line count at the predicted size
            # (a predicted box can be one line short; shrinking to it loses the designer's size).
            lines = max(1, measure(el, int(el['font_size']), float(el['width'])).lines)
            budget = max(h * 1.6, size * 1.3, lines * size * float(el.get('line_height') or 1.2) * 1.1)
            fit = fit_text(el, w, budget, size, fs_min=min(size, _floor(A, el)))
            if fit.overflow or fit.height > tH:
                fit = fit_text(el, w, tH * 0.9, size, fs_min=1)
            top, bottom = y, y + h
            h = fit.height
            y = {'top': top, 'bottom': bottom - h}.get(anchor.get(el['id']), (top + bottom - h) / 2)
            out['font_size'] = int(fit.font_size)
            # Copy stays fully on the canvas, inside its safe margin when it fits.
            x = _clamp(x, m, tW - m - w, min(max(x, 0.0), tW - w))
            y = _clamp(y, m, tH - m - h, min(max(y, 0.0), tH - h))
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
    """Up to k learned candidates as (tag, design); sample 0 is the model's greedy decode.

    The greedy layout also comes in a `bound` form whose copy stays with any graphic it sat on
    (photos and panels included). The engine's checks decide whether it is needed.
    """
    if not available() or len(source['elements']) > MAX_ELEMENTS:
        return []
    recs = records(source, A)
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    preds = predict(recs, W, H, float(target['width']), float(target['height']), k=k, temperature=temperature, seed=seed)
    out = [(str(i), deoverlap(source, materialize(source, target, A, recs, p), A)) for i, p in enumerate(preds)]
    if preds and backed(source, True) != backed(source):
        out.insert(1, ('0_bound', deoverlap(source, materialize(source, target, A, recs, preds[0], loose=True), A, loose=True)))
    return out + adapted(source, target, A, recs)


def adapted(source, target, A, recs, k=2):
    """Extra candidates for extreme strips from the strip adapter (training/retarget/adapter.py):
    its greedy layout and one sample, tagged a0/a1. The base model's candidates are unchanged and
    the engine picks among all of them, so the adapter can only add options."""
    from .infer import load, strip_adapter, strip_gate, weights_path
    g = strip_gate(float(target['width']), float(target['height']))
    path = strip_adapter()
    if not g or path is None:
        return []
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    preds = predict(recs, W, H, float(target['width']), float(target['height']), k=k, model=load(weights_path(), path), gate=g)
    return [(f'a{i}', deoverlap(source, materialize(source, target, A, recs, p), A)) for i, p in enumerate(preds)]


def candidates(source, target, A, options, events):
    """(template, name, description, design) proposals, as the pipeline and providers assemble them."""
    import os
    if not options.get('learned_retarget', os.environ.get('CHAMELEON_LEARNED_RETARGET', '1') != '0') \
            or source['canvas'] == target:
        return []
    try:
        out = [(f'learned_{tag}', 'Learned layout', 'Predicted by the retargeting model trained on designer resizes', d)
               for tag, d in propose(source, target, A, k=int(options.get('learned_samples', 6)))]
    except (ValueError, OSError, KeyError) as exc:
        events.append({'learned_retarget': f'unavailable: {exc}'})
        return []
    if postproc(options) == 'solver' and out:
        out = solved(source, target, A, out[0][3], events, options) + out
    return [(t, n, d, keep_bleed(source, design)) for t, n, d, design in out]



def postproc(options=None):
    """Post-processing for learned layouts: 'solver' (constrained decoding and appeal-ranked
    selection, the default) or 'off' (today's materialize, fit and de-overlap only). Set per
    request with options['postproc'] or for the process with CHAMELEON_POSTPROC."""
    import os
    mode = str((options or {}).get('postproc') or os.environ.get('CHAMELEON_POSTPROC') or 'solver')
    return 'solver' if mode == 'solver' else 'off'


def keep_bleed(source, design):
    """A layer that showed past the opaque layer above it in the source (often a background photo
    peeking out at one edge) must not end up fully covered: that deletes it in effect, which the
    engine's rubric rejects. Re-open it on the side it showed in the source, inside the canvas:
    grow it to the canvas edge if that is enough, else inset the covering layer by a few pixels."""
    from ..rubric import _contains, _covered, _opaque, _order
    from ..canonical import rotated_extent
    newly = _covered(design) - _covered(source)
    if not newly:
        return design
    import copy
    src = {e['id']: e for e in source['elements']}
    sW, sH = source['canvas']['width'], source['canvas']['height']
    W, H = design['canvas']['width'], design['canvas']['height']
    clip = lambda b, w, h: (max(b.x, 0), max(b.y, 0), min(b.x1, w), min(b.y1, h))
    out = copy.deepcopy(design)
    by = {e['id']: e for e in out['elements']}
    order = _order(out)
    for i in newly:
        e = by.get(i)
        if e is None or i not in src or e.get('rotation'):
            continue
        over = [c for c in out['elements'] if c is not e and order[c['id']] > order[i] and _opaque(c)
                and _contains(rotated_extent(c), rotated_extent(e)) and c['id'] in src and not c.get('rotation')]
        if not over:
            continue
        # the side where this layer showed past its covers inside the source canvas
        sl, st, sr, sb = clip(rotated_extent(src[i]), sW, sH)
        cl = [clip(rotated_extent(src[c['id']]), sW, sH) for c in over]
        excess = {'l': min(c[0] for c in cl) - sl, 't': min(c[1] for c in cl) - st,
                  'r': sr - max(c[2] for c in cl), 'b': sb - max(c[3] for c in cl)}
        side = max(excess, key=excess.get)
        if excess[side] <= 0:
            continue
        horiz = side in 'lr'
        pad = max(2.0, round(0.01 * (W if horiz else H)))
        k0, k1, lim = ('x', 'width', W) if horiz else ('y', 'height', H)
        lo = min(c[k0] for c in over); hi = max(c[k0] + c[k1] for c in over)
        if side in 'lt' and lo - pad >= 0:
            e[k1] += e[k0] - (lo - pad); e[k0] = lo - pad
        elif side in 'rb' and hi + pad <= lim:
            e[k1] = hi + pad - e[k0]
        else:
            for c in over:
                if c.get('type') == 'image' and c.get('fit') not in ('cover', None):
                    continue  # a fill-fit image would be distorted by an inset
                if side in 'lt':
                    cut = max(0.0, e[k0] + pad - c[k0]); c[k0] += cut; c[k1] -= cut
                else:
                    c[k1] = min(c[k1], e[k0] + e[k1] - pad - c[k0])
    return out


SOLVER_PROFILES = {'balanced': {}, 'bold': {'headline': 0.6, 'margin': 0.25, 'balance': 0.1}}


def solved(source, target, A, warm, events, options=None):
    """Constrained decoding of the model's greedy distribution (engine/retarget/solver.py) under each
    weight profile, warm-started from today's greedy layout, or for extreme strips from the
    reading-order composer (which is itself a candidate). Tagged learned_s0_<profile> and
    learned_strip; the engine ranks the ones that pass every check by appeal."""
    from .infer import predict_dist
    from .solver import Problem, solve
    from .strip import compose
    options = options or {}
    out = []
    try:
        recs = records(source, A)
        W, H = float(source['canvas']['width']), float(source['canvas']['height'])
        boxes, probs, _ = predict_dist(recs, W, H, float(target['width']), float(target['height']), k=1)
        P = Problem(source, target, A, recs, _priors())
        if P.strip:
            warm = compose(P)
            out.append(('learned_strip', 'Strip layout', 'The content in reading order along the strip', warm))
        # One shared deadline across the profiles, so a hard layout cannot hold up the request.
        import time
        deadline = time.monotonic() + float(options.get('solver_budget', 5.0))
        sols = [(name, solve(P, boxes[0], probs[0], weights=wts, time_limit=float(options.get('solver_seconds', 1.5)), hint=warm,
                             workers=int(options.get('solver_workers', 8)), deadline=deadline))
                for name, wts in SOLVER_PROFILES.items()]
        for name, sol in sols:
            if sol is not None and sol.design is not None:
                out.append((f'learned_s0_{name}', 'Learned layout (solved)', 'The model\'s most likely layout that satisfies every check', sol.design))
                events.append({'learned_solver': {'profile': name, 'status': sol.status, 'notes': sol.notes[-3:]}})
            else:
                events.append({'learned_solver': {'profile': name, 'status': 'no solution'}})
        out += _solved_adapter(P, recs, W, H, target, warm, events, options) if P.strip else []
    except Exception as exc:  # the solver is an addition: today's candidates stand without it
        events.append({'learned_solver': f'failed: {type(exc).__name__}: {exc}'})
    return out


def _solved_adapter(P, recs, W, H, target, warm, events, options):
    """For strips, one more solve under the strip adapter's distribution (in-distribution for
    strips, so its model cost is trusted more), with its own time slice so the profiles above keep
    theirs. Tagged learned_s0_adapter."""
    import time
    from .infer import load, predict_dist, strip_adapter, strip_gate, weights_path
    from .solver import solve
    path, g = strip_adapter(), strip_gate(float(target['width']), float(target['height']))
    if path is None or not g:
        return []
    boxes, probs, _ = predict_dist(recs, W, H, float(target['width']), float(target['height']), k=1,
                                   model=load(weights_path(), path), gate=g)
    deadline = time.monotonic() + float(options.get('adapter_budget', 2.5))
    sol = solve(P, boxes[0], probs[0], weights={'strip_model': float(options.get('adapter_trust', 1.0))},
                time_limit=float(options.get('solver_seconds', 1.5)), hint=warm, workers=int(options.get('solver_workers', 8)),
                deadline=deadline)
    if sol is None or sol.design is None:
        events.append({'learned_solver': {'profile': 'adapter', 'status': 'no solution'}})
        return []
    events.append({'learned_solver': {'profile': 'adapter', 'status': sol.status, 'notes': sol.notes[-3:]}})
    return [('learned_s0_adapter', 'Learned layout (solved, strip adapter)', 'The strip adapter\'s most likely layout that satisfies every check', sol.design)]


_PRIORS = None


def _priors():
    """Designer priors (margins, headline scale, ...) measured on Crello's train split."""
    global _PRIORS
    if _PRIORS is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / 'models' / 'designer_priors.json'
        _PRIORS = json.loads(p.read_text()) if p.exists() else {}
    return _PRIORS
