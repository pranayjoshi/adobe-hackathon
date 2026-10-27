"""Shared machinery for the large-scale evaluation (reads engine/, never edits it).

Data lives under work/ (git-ignored): work/retarget-data (Crello layouts), work/retarget-scale
(fetched layer images, rawrows.pkl) -- rebuilt by training/retarget/fetch_crello.py and the scale run."""
import copy
import math
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (tools/postproc -> repo)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
SC = ROOT / 'work/retarget-scale'
IMG_DIRS = [SC / 'images', ROOT / 'work/retarget-data/images']

import numpy as np

PANEL = {'1080x1080': (1080, 1080), '1080x1920': (1080, 1920), '1920x1080': (1920, 1080),
         '1200x628': (1200, 628), '1600x400': (1600, 400), '160x600': (160, 600),
         '300x250': (300, 250), '728x90': (728, 90)}


def bucket(la):
    la = abs(la)
    return 'b0_<0.3' if la < 0.3 else 'b1_0.3-0.7' if la < 0.7 else 'b2_0.7-1.1' if la < 1.1 else 'b3_>1.1'


def free_targets(W, H):
    """3 panel targets: largest log-aspect change, then closest to 0.7, then closest to 0.3 (distinct);
    targets within 0.05 of the source log-aspect are excluded."""
    a = math.log(W / H)
    cands = [(k, abs(a - math.log(w / h))) for k, (w, h) in PANEL.items() if abs(a - math.log(w / h)) >= 0.05]
    pick = [max(cands, key=lambda c: c[1])]
    for goal in (0.7, 0.3):
        rest = [c for c in cands if c[0] not in [p[0] for p in pick]]
        if rest:
            pick.append(min(rest, key=lambda c: abs(c[1] - goal)))
    return [(k, d, PANEL[k]) for k, d in pick]


class Env:
    """Per-process lazy state."""
    def __init__(self):
        from training.retarget.data import load
        self.designs, self.pairs = load(ROOT / 'work/retarget-data')
        self.rows = pickle.load(open(SC / 'rawrows.pkl', 'rb'))
        from engine.analysis import analyze
        from engine.retarget import propose as P
        from engine.retarget.features import MAX_ELEMENTS
        self.analyze, self.P, self.MAXEL = analyze, P, MAX_ELEMENTS
        self._src = (None, None)

    def image_path(self, did):
        for d in IMG_DIRS:
            p = d / f'{did}.pkl'
            if p.exists():
                return p
        return None

    def source(self, idx):
        """-> (engine design, analysis, recs) or raises ValueError(reason)."""
        if self._src[0] == idx:
            return self._src[1]
        from training.retarget.visual import engine_source
        D = self.designs[idx]
        p = self.image_path(D['id'])
        if p is None:
            raise ValueError('no_image_file')
        images = pickle.load(open(p, 'rb'))
        try:
            key, src = engine_source(self.rows[D['id']], images)
        except ValueError as exc:
            msg = str(exc)
            raise ValueError('convert:' + ('missing_image' if 'Missing original image' in msg else 'missing_font' if 'Missing font' in msg
                                           else 'invalid_geometry' if 'Invalid geometry' in msg else msg[:60]))
        except Exception as exc:  # noqa
            raise ValueError('convert_exc:' + type(exc).__name__ + ':' + str(exc)[:60])
        if len(src['elements']) > self.MAXEL:
            raise ValueError('too_many_elements')
        A = self.analyze(src)
        recs = self.P.records(src, A)
        self._src = (idx, (src, A, recs))
        return self._src[1]


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ---------------------------------------------------------------- raw model geometry
def raw_design(source, target, recs, pred):
    """Model boxes with only the hard shape rules of evaluate.apply_pred (graphics aspect-locked,
    covers fill the canvas). No clamping, margins, fitting or overlap removal."""
    from engine.retarget.features import TEXT
    tW, tH = float(target['width']), float(target['height'])
    by = {r['id']: (r, p) for r, p in zip(recs, pred)}
    els = []
    for el in source['elements']:
        r, p = by[el['id']]
        out = copy.deepcopy(el)
        if r['cover']:
            x, y, w, h = 0.0, 0.0, tW, tH
        elif r['type'] == TEXT:
            x, y, w, h = p['x'], p['y'], p['w'], p['h']
            out['font_size'] = float(p['font_size'])
        else:
            a = r['w'] / r['h']; area = p['w'] * p['h']
            w, h = math.sqrt(area * a), math.sqrt(area / a)
            x, y = p['x'] + p['w'] / 2 - w / 2, p['y'] + p['h'] / 2 - h / 2
        out.update(x=float(x), y=float(y), width=float(w), height=float(h))
        els.append(out)
    return {'canvas': {'width': int(tW), 'height': int(tH)}, 'elements': els}


# ---------------------------------------------------------------- reference scoring
def ref_records(B, mapping):
    return [B['els'][mapping[i]] for i in sorted(mapping)]


def score_vs_ref(layout, ref, tW, tH, keys, extra=True):
    """evaluate.metrics plus per-element-type and per-role aggregates."""
    from training.retarget.evaluate import metrics, iou, is_cover
    m = metrics(layout, ref, tW, tH)
    if m is None:
        return None
    m = {k: float(v) for k, v in m.items()}
    if extra:
        grp = {}
        for a, b, (typ, role) in zip(layout, ref, keys):
            if is_cover(b, tW, tH):
                continue
            ce = abs((a['x'] + a['w'] / 2 - b['x'] - b['w'] / 2) / tW) / 2 + abs((a['y'] + a['h'] / 2 - b['y'] - b['h'] / 2) / tH) / 2
            se = abs(math.log(max(a['w'] * a['h'], 1e-3) / max(b['w'] * b['h'], 1e-3)))
            fe = None
            if typ == 'text' and (a.get('font_size') or 0) > 0 and (b.get('font_size') or 0) > 0:
                fe = abs(math.log(a['font_size'] / b['font_size']))
            for g in ('t:' + typ, 'r:' + role):
                s = grp.setdefault(g, [0, 0.0, 0.0, 0.0, 0, 0.0])
                s[0] += 1; s[1] += iou(a, b); s[2] += ce; s[3] += se
                if fe is not None:
                    s[4] += 1; s[5] += fe
        m['groups'] = {g: [s[0], round(s[1], 4), round(s[2], 4), round(s[3], 4), s[4], round(s[5], 4)] for g, s in grp.items()}
    return m


# ---------------------------------------------------------------- reference-free checks
def _eb(e):
    return float(e['x']), float(e['y']), float(e['width']), float(e['height'])


def _largest_empty(occ):
    n = occ.shape[0]
    best = 0
    heights = np.zeros(occ.shape[1], int)
    for r in range(n):
        heights = np.where(occ[r], 0, heights + 1)
        stack = []
        for c in range(occ.shape[1] + 1):
            h = heights[c] if c < occ.shape[1] else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, hh = stack.pop()
                best = max(best, hh * (c - s)); start = s
            stack.append((start, h))
    return best / occ.size


def _content_boxes(d):
    from engine.quality import ink as inkbox
    from engine.canonical import rotated_extent
    cw, ch = float(d['canvas']['width']), float(d['canvas']['height'])
    out = []
    for e in d['elements']:
        b = rotated_extent(e)
        if b.w * b.h >= 0.9 * cw * ch:
            continue
        if e['type'] == 'text':
            try:
                b = inkbox(e)
            except Exception:
                pass
        out.append((b.x, b.y, b.w, b.h))
    return out


def _content_stats(d):
    cw, ch = float(d['canvas']['width']), float(d['canvas']['height'])
    boxes = _content_boxes(d)
    if not boxes:
        return {'empty_rect': 1.0, 'cx': 0.0, 'cy': 0.0, 'bbox_share': 0.0, 'coverage': 0.0}
    x0 = max(0, min(b[0] for b in boxes)); y0 = max(0, min(b[1] for b in boxes))
    x1 = min(cw, max(b[0] + b[2] for b in boxes)); y1 = min(ch, max(b[1] + b[3] for b in boxes))
    n = 40; occ = np.zeros((n, n), bool)
    for x, y, w, h in boxes:
        a0, a1 = max(0, int(x / cw * n)), min(n, int(math.ceil((x + w) / cw * n)))
        b0, b1 = max(0, int(y / ch * n)), min(n, int(math.ceil((y + h) / ch * n)))
        if a1 > a0 and b1 > b0:
            occ[b0:b1, a0:a1] = True
    return {'empty_rect': _largest_empty(occ), 'cx': (x0 + x1) / 2 / cw - 0.5, 'cy': (y0 + y1) / 2 / ch - 0.5,
            'bbox_share': max(0, (x1 - x0)) * max(0, (y1 - y0)) / (cw * ch), 'coverage': float(occ.mean())}


def _texts(d):
    return [e for e in d['elements'] if e['type'] == 'text' and str(e.get('content', '')).strip()]


def _reading_order(els, H):
    band = 0.08 * H
    key = lambda e: (int((e['y'] + e['height'] / 2) // band), e['x'] + e['width'] / 2)
    return [e['id'] for e in sorted(els, key=key)]


def _discord(a, b):
    pos = {i: k for k, i in enumerate(b)}
    n = len(a); tot = n * (n - 1) // 2
    if tot == 0:
        return 0, 0
    d = sum(1 for i in range(n) for j in range(i + 1, n) if pos[a[i]] > pos[a[j]])
    return d, tot


def _align_pairs(texts, W):
    """Pairs of texts sharing a left / right / centre edge within 1% of the width, on different rows."""
    out = set()
    tol = 0.01 * W
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if abs((a['y'] + a['height'] / 2) - (b['y'] + b['height'] / 2)) < 0.3 * min(a['height'], b['height']):
                continue
            ax = (a['x'], a['x'] + a['width'], a['x'] + a['width'] / 2)
            bx = (b['x'], b['x'] + b['width'], b['x'] + b['width'] / 2)
            for k, nm in enumerate(('L', 'R', 'C')):
                if abs(ax[k] - bx[k]) <= tol:
                    out.add((a['id'], b['id'], nm))
    return out


def style_stats(d):
    """Designer-prior style quantities on an engine-format design (same definitions as priors.py)."""
    from engine.canonical import rotated_extent
    W, H = float(d['canvas']['width']), float(d['canvas']['height']); short = min(W, H)
    tx = [e for e in d['elements'] if e['type'] == 'text' and str(e.get('content', '')).strip()]
    out = {}
    if tx:
        out['text_margin_min'] = min(min(e['x'], e['y'], W - e['x'] - e['width'], H - e['y'] - e['height']) for e in tx) / short
        out['text_margin_each'] = [min(e['x'], e['y'], W - e['x'] - e['width'], H - e['y'] - e['height']) / short for e in tx]
        hl = max(tx, key=lambda e: e['font_size'])
        out['headline_fs_over_short'] = hl['font_size'] / short
        out['headline_h_over_short'] = hl['height'] / short
        out['text_area_share'] = sum(e['width'] * e['height'] for e in tx) / (W * H)
        out['min_fs_over_short'] = min(e['font_size'] for e in tx) / short
        fs = sorted((e['font_size'] for e in tx), reverse=True)
        if len(fs) >= 2:
            out['headline_over_second_font'] = fs[0] / max(fs[1], 1e-6)
    gaps = []
    for a in tx:
        best = None
        for b in tx:
            if b is a or b['y'] <= a['y']:
                continue
            ox = min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x'])
            if ox < 0.3 * min(a['width'], b['width']):
                continue
            g = b['y'] - (a['y'] + a['height'])
            if best is None or g < best:
                best = g
        if best is not None:
            gaps.append(best / max(a['font_size'], 1e-6))
    out['stack_gap_over_font'] = gaps
    if len(tx) >= 2:
        sx = float(np.std([e['x'] + e['width'] / 2 for e in tx])) / W; sy = float(np.std([e['y'] + e['height'] / 2 for e in tx])) / H
        out['text_spread'] = [sx, sy]
        out['arrangement'] = 'stack' if sy > 1.3 * sx else 'row' if sx > 1.3 * sy else 'mixed'
    tol = 0.01 * W; al = set()
    for i in range(len(tx)):
        for j in range(i + 1, len(tx)):
            a, b = tx[i], tx[j]
            if abs((a['y'] + a['height'] / 2) - (b['y'] + b['height'] / 2)) < 0.3 * min(a['height'], b['height']):
                continue
            ax = (a['x'], a['x'] + a['width'], a['x'] + a['width'] / 2); bx = (b['x'], b['x'] + b['width'], b['x'] + b['width'] / 2)
            if any(abs(ax[k] - bx[k]) <= tol for k in range(3)):
                al |= {i, j}
    if len(tx) >= 2:
        out['text_aligned_share'] = len(al) / len(tx)
    cst = _content_stats(d)
    out.update(content_bbox_share=cst['bbox_share'], coverage=cst['coverage'], empty_rect=cst['empty_rect'])
    # text over a backing: share and padding in font units
    grs = [e for e in d['elements'] if e['type'] != 'text' and e['width'] * e['height'] <= 0.25 * W * H]
    pads, nb = [], 0
    for e in tx:
        best = None
        for b in grs:
            if b.get('z_index', 0) >= e.get('z_index', 0) or b['width'] * b['height'] < e['width'] * e['height']:
                continue
            ix = max(0, min(e['x'] + e['width'], b['x'] + b['width']) - max(e['x'], b['x']))
            iy = max(0, min(e['y'] + e['height'], b['y'] + b['height']) - max(e['y'], b['y']))
            if ix * iy / max(e['width'] * e['height'], 1e-6) >= 0.75 and (best is None or b['width'] * b['height'] < best['width'] * best['height']):
                best = b
        if best:
            nb += 1
            pads.append(min(e['x'] - best['x'], best['x'] + best['width'] - e['x'] - e['width'], e['y'] - best['y'], best['y'] + best['height'] - e['y'] - e['height']) / max(e['font_size'], 1e-6))
    out['n_text_backed'] = nb; out['backing_pad_min_over_font'] = pads
    return out


def source_info(src, A):
    """Cached per-source quantities used by every output check."""
    from engine.pipeline import _content_share
    from engine.quality import composite, text_contrast
    from engine.retarget.deoverlap import backed
    info = {'content_share': _content_share(src), 'W': float(src['canvas']['width']), 'H': float(src['canvas']['height'])}
    try:
        im, _ = composite(src)
        info['contrast'], _ = text_contrast(src, im)
    except Exception:
        info['contrast'] = {}
    try:
        info['backed'] = backed(src)
    except Exception:
        info['backed'] = []
    from engine.retarget.propose import _units
    info['groups'] = [list(u) for u in _units(src, A)]
    from engine.canonical import rotated_extent
    info['off'] = [e['id'] for e in src['elements'] if (lambda b: b.x < -0.5 or b.y < -0.5 or b.x1 > info['W'] + .5 or b.y1 > info['H'] + .5)(rotated_extent(e))]
    info['content'] = _content_stats(src)
    return info


def check(src, D, A, si, pred_fs=None):
    """All reference-free measurements for output design D of source src. Raw values."""
    from engine.invariants import validate
    from engine.quality import collisions
    from engine.typography import measure
    from engine.blocks import FLOORS
    from engine.canonical import rotated_extent
    from engine.pipeline import _new_occlusion, _content_share
    out = {}
    W, H = si['W'], si['H']
    tW, tH = float(D['canvas']['width']), float(D['canvas']['height'])
    short = min(tW, tH); sshort = min(W, H)
    inv = validate(src, D)
    out['inv_fail'] = [c['name'] for c in inv if c['severity'] == 'error' and not c['passed']]
    out['inv_detail'] = {c['name']: c['detail'][:80] for c in inv if not c['passed']}
    try:
        col = collisions(src, D, A)
        out['coll_n'] = len(col)
        out['coll_worst'] = max([c['fraction'] for c in col], default=0.0)
        out['coll'] = sorted(({'f': c['fraction'], 'roles': sorted({A.roles.get(i, '?') for i in c['ids']})} for c in col),
                             key=lambda r: -r['f'])[:4]
    except Exception as exc:
        out['coll_err'] = str(exc)[:80]
    try:
        from engine.scene import scene_defects
        sd = scene_defects(src, D)
        out['unsupported_text'] = sd['unsupported_text']
        out['scene_displacement'] = sd['scene_displacement']
    except Exception as exc:
        out['scene_err'] = str(exc)[:80]
    try:
        out['occluded'] = _new_occlusion(src, D)
    except Exception as exc:
        out['occluded'] = []
    cs = _content_share(D)
    out['content_share'] = cs
    out['content_share_ratio'] = cs / max(si['content_share'], 1e-6)
    by_s = {e['id']: e for e in src['elements']}
    tx = _texts(D)
    ov, below, small, fsr_src, fsr_pred, below_roles, ov_detail = [], [], [], [], [], [], []
    for e in tx:
        fs = e['font_size']; s = by_s[e['id']]
        try:
            f = measure(e, int(fs), max(1, e['width']))
            if f.overflow or f.height > e['height'] + .5:
                ov.append(e['id'])
                ov_detail.append({'role': A.roles.get(e['id'], 'body'), 'fs': fs, 'src_fs': s['font_size'], 'floor': FLOORS.get(A.roles.get(e['id'], 'body'), 8),
                                  'need_h': f.height, 'h': e['height'], 'w': e['width'], 'ovf': bool(f.overflow)})
        except Exception:
            pass
        role = A.roles.get(e['id'], 'body')
        if fs < FLOORS.get(role, 8):
            below.append(e['id']); below_roles.append(role)
        if fs < 7:
            small.append(e['id'])
        fsr_src.append(fs / max(s['font_size'], 1))
        if pred_fs and e['id'] in pred_fs:
            fsr_pred.append(fs / max(pred_fs[e['id']], 1))
    out.update(n_text=len(tx), n_el=len(D['elements']), text_overflow=ov, overflow_detail=ov_detail[:4], below_floor_roles=below_roles,
               min_fs_short=(min(e['font_size'] for e in tx) / short) if tx else None, text_below_floor=below, text_small=small,
               font_ratio_src_med=float(np.median(fsr_src)) if fsr_src else None,
               font_ratio_src_min=float(min(fsr_src)) if fsr_src else None,
               font_ratio_pred_min=float(min(fsr_pred)) if fsr_pred else None,
               font_ratio_pred_med=float(np.median(fsr_pred)) if fsr_pred else None,
               n_fit_shrunk=sum(1 for r in fsr_pred if r < 0.9))
    viol, gaps = [], []
    for e in tx:
        s = by_s[e['id']]
        gt = min(e['x'], e['y'], tW - e['x'] - e['width'], tH - e['y'] - e['height'])
        gs = min(s['x'], s['y'], W - s['x'] - s['width'], H - s['y'] - s['height'])
        gaps.append(gt / short)
        if gt < 0.04 * short - 0.5 and gs / sshort >= 0.04:
            viol.append(e['id'])
    out['margin_viol'] = viol
    out['min_text_gap'] = min(gaps) if gaps else None
    out['off_canvas'] = [e['id'] for e in D['elements'] if (lambda b: b.x < -0.5 or b.y < -0.5 or b.x1 > tW + .5 or b.y1 > tH + .5)(rotated_extent(e))]
    out['off_canvas_new'] = [i for i in out['off_canvas'] if i not in si['off']]
    out['hidden'] = [[e['type'], A.roles.get(e['id'], '?'), round(e['width'], 2), round(e['height'], 2),
                      round(by_s[e['id']]['width'], 1), round(by_s[e['id']]['height'], 1)] for e in D['elements'] if e['width'] < 1 or e['height'] < 1]
    stx = [by_s[e['id']] for e in tx]
    if len(tx) >= 2:
        d, tot = _discord(_reading_order(stx, H), _reading_order(tx, tH))
        out['order_discord'] = d / tot
        fl = hp = hflat = 0
        for i in range(len(tx)):
            for j in range(len(tx)):
                if i == j:
                    continue
                if stx[i]['font_size'] >= 1.25 * stx[j]['font_size']:
                    hp += 1
                    ta, tb = tx[i]['font_size'], tx[j]['font_size']
                    if ta < tb:
                        fl += 1
                    elif ta / max(tb, 1) < 1.1:
                        hflat += 1
        out.update(hier_pairs=hp, hier_flips=fl, hier_flat=hflat)
        sa_ = _align_pairs(stx, W)
        ta_ = _align_pairs(tx, tW)
        out.update(align_pairs=len(sa_), align_kept=len(sa_ & ta_))
    gl, back_lost, nb = [], 0, 0
    dby = {e['id']: e for e in D['elements']}
    for ids in si['groups']:
        ids = [i for i in ids if i in dby]
        if len(ids) < 2:
            continue

        def rel(get):
            bs = [get(i) for i in ids]
            x0 = min(b[0] for b in bs); y0 = min(b[1] for b in bs)
            x1 = max(b[0] + b[2] for b in bs); y1 = max(b[1] + b[3] for b in bs)
            uw, uh = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
            return [((b[0] + b[2] / 2 - (x0 + x1) / 2) / uw, (b[1] + b[3] / 2 - (y0 + y1) / 2) / uh) for b in bs], uw * uh
        rs, areas = rel(lambda i: _eb(by_s[i])); rt, areat = rel(lambda i: _eb(dby[i]))
        err = float(np.mean([abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(rs, rt)]) / 2)
        sc = np.median([math.sqrt(_eb(dby[i])[2] * _eb(dby[i])[3] / max(_eb(by_s[i])[2] * _eb(by_s[i])[3], 1e-3)) for i in ids])
        gl.append((err, math.sqrt(areat / areas) / max(sc, 1e-3), len(ids)))
    out['n_groups'] = len(gl)
    out['group_err_max'] = max([g[0] for g in gl], default=0.0)
    out['group_err_mean'] = float(np.mean([g[0] for g in gl])) if gl else 0.0
    out['groups_broken'] = sum(1 for g in gl if g[0] > 0.25)
    out['group_disp_max'] = max([g[1] for g in gl], default=1.0)
    for g in si['backed']:
        els = [by_s[i] for i in g if i in by_s]
        host = max((e for e in els if e['type'] != 'text'), key=lambda e: e['width'] * e['height'], default=None)
        if host is None:
            continue
        for e in els:
            if e['type'] != 'text':
                continue
            nb += 1
            h2, e2 = dby[host['id']], dby[e['id']]
            ix = max(0.0, min(e2['x'] + e2['width'], h2['x'] + h2['width']) - max(e2['x'], h2['x']))
            iy = max(0.0, min(e2['y'] + e2['height'], h2['y'] + h2['height']) - max(e2['y'], h2['y']))
            if ix * iy / max(e2['width'] * e2['height'], 1e-6) < 0.75:
                back_lost += 1
    out.update(backed_text=nb, backed_text_lost=back_lost)
    cst = _content_stats(D)
    out.update(empty_rect=cst['empty_rect'], content_cx=cst['cx'], content_cy=cst['cy'], bbox_share=cst['bbox_share'],
               coverage=cst['coverage'], src_empty_rect=si['content']['empty_rect'], src_coverage=si['content']['coverage'])
    if tx:
        hl = max(tx, key=lambda e: e['font_size'])
        out.update(hl_fs_short=hl['font_size'] / short, hl_h_short=hl['height'] / short, hl_w_frac=hl['width'] / tW,
                   hl_fs_src_short=by_s[hl['id']]['font_size'] / sshort,
                   text_area_share=sum(e['width'] * e['height'] for e in tx) / (tW * tH))
    out['style'] = style_stats(D)
    try:
        from engine.quality import composite, text_contrast
        im, _ = composite(D)
        ratios, busy = text_contrast(D, im)
        src_r = si['contrast']
        out['contrast_min'] = min(ratios.values()) if ratios else None
        out['contrast_low'] = [i for i, v in ratios.items() if v < 3]
        out['contrast_regress'] = [i for i, v in ratios.items() if v < 3 and v < .6 * src_r.get(i, 21)]
        out['src_contrast_min'] = min(src_r.values()) if src_r else None
    except Exception as exc:
        out['contrast_err'] = str(exc)[:80]
    return out
