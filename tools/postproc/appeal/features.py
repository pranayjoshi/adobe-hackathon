"""Reference-free design features for the appeal scorer (no designer layout is ever used).

Composition (balance, fill, empty space, margins), typography (sizes, hierarchy, line breaks),
structure kept from the source (alignment, order, groups, hosts), rule-level health (collisions,
overflow, contrast), and the engine rubric's dimension scores.
"""
import math

import numpy as np

ROLE_W = {'headline': 3.0, 'subhead': 2.0, 'cta': 2.0, 'body': 1.0, 'legal': 0.5}


def _visual_balance(design):
    cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    mx = my = mt = 0.0
    for e in design['elements']:
        w, h = max(float(e['width']), 0), max(float(e['height']), 0)
        if w >= .95 * cw and h >= .95 * ch:
            continue
        m = w * h * (2.0 if e['type'] == 'text' else 1.0)
        mx += m * (float(e['x']) + w / 2); my += m * (float(e['y']) + h / 2); mt += m
    if mt <= 0:
        return 0.0, 0.0
    return mx / mt / cw - .5, my / mt / ch - .48


def _line_stats(design):
    from engine.retarget.solver import STOPWORDS, _wrap_lines
    widows = stops = 0
    rags = []
    for e in design['elements']:
        if e['type'] != 'text' or not str(e.get('content', '')).strip():
            continue
        try:
            lines = _wrap_lines(e, int(e['font_size']), float(e['width']))
        except Exception:
            continue
        words = [w for l in lines for w in l.split()]
        if len(words) >= 3 and len(lines) >= 2 and len(lines[-1].split()) == 1:
            widows += 1
        for l in lines[:-1]:
            ws = l.split()
            if ws and ws[-1].lower().strip(',.:;') in STOPWORDS:
                stops += 1
        if len(lines) >= 2:
            lens = [len(l.strip()) for l in lines]
            rags.append(float(np.std(lens) / max(np.mean(lens), 1)))
    return widows, stops, (float(np.mean(rags)) if rags else 0.0)


def _gaps(design):
    """Vertical gaps between consecutive stacked copy, in font heights: spread = rhythm."""
    tx = sorted((e for e in design['elements'] if e['type'] == 'text'), key=lambda e: e['y'])
    g = []
    for a, b in zip(tx, tx[1:]):
        if min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x']) > 0:
            g.append((b['y'] - a['y'] - a['height']) / max(a['font_size'], 1))
    return (float(np.std(g)) if len(g) >= 2 else 0.0), (float(np.median(g)) if g else 0.0), len(g)


def features(src, design, A, si):
    """Feature dict for design as a resize of src (si: harness.source_info(src, A))."""
    import harness
    tgt = design['canvas']
    c = harness.check(src, design, A, si, None)
    ok, reasons, rub = harness.eligible(src, tgt, design, A)
    cw, ch = float(tgt['width']), float(tgt['height'])
    short = min(cw, ch)
    f = {}
    for k in ('coll_n', 'coll_worst', 'content_share', 'content_share_ratio', 'n_text', 'n_el', 'min_fs_short',
              'font_ratio_src_med', 'font_ratio_src_min', 'n_fit_shrunk', 'min_text_gap', 'order_discord', 'hier_pairs',
              'hier_flips', 'hier_flat', 'align_pairs', 'align_kept', 'group_err_max', 'groups_broken', 'backed_text_lost',
              'empty_rect', 'content_cx', 'content_cy', 'bbox_share', 'coverage', 'src_empty_rect', 'src_coverage',
              'hl_fs_short', 'hl_h_short', 'hl_w_frac', 'hl_fs_src_short', 'text_area_share', 'contrast_min', 'src_contrast_min'):
        v = c.get(k)
        f[k] = float(v) if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else float('nan')
    for k in ('inv_fail', 'unsupported_text', 'scene_displacement', 'occluded', 'text_overflow', 'text_below_floor',
              'text_small', 'margin_viol', 'off_canvas_new', 'hidden', 'contrast_low', 'contrast_regress'):
        f['n_' + k] = float(len(c.get(k) or []))
    f['align_keep_rate'] = f['align_kept'] / f['align_pairs'] if f['align_pairs'] and f['align_pairs'] > 0 else 1.0
    f['coverage_gain'] = f['coverage'] - f['src_coverage']
    f['empty_gain'] = f['empty_rect'] - f['src_empty_rect']
    f['hl_scale'] = f['hl_fs_short'] / f['hl_fs_src_short'] if f['hl_fs_src_short'] and f['hl_fs_src_short'] > 0 else float('nan')
    bx, by = _visual_balance(design)
    f['balance_x'], f['balance_y'], f['balance'] = bx, by, math.hypot(bx, by)
    f['widows'], f['stop_breaks'], f['rag'] = _line_stats(design)
    f['gap_spread'], f['gap_median'], f['n_gaps'] = _gaps(design)
    f['aspect_log'] = math.log(cw / ch)
    f['aspect_change'] = abs(math.log((float(src['canvas']['width']) / float(src['canvas']['height'])) / (cw / ch)))
    f['eligible'] = float(ok)
    f['n_veto'] = float(len(reasons))
    f['rubric_overall'] = float(rub.get('overall') or float('nan'))
    for k, v in (rub.get('scores') or {}).items():
        f['rubric_' + k] = float(v)
    return f
