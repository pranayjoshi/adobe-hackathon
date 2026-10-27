"""Order-preserving overlap removal for learned layouts.

The engine's generic repair pushes colliding blocks wherever there is room, which can scatter
a learned composition (a title split across the canvas, two lines swapping order). Here each
analysis unit (the same units the engine's collision check uses) only translates, by the
smallest total area-weighted distance, subject to:

  * every unit pair that collides keeps the order the source had on the axis that separated
    them there (above stays above, left stays left), with a small gap;
  * an overlap the source already had (text on a shape) may stay, up to its source amount;
  * every unit stays inside the canvas.

Collisions created by a solve join the constraint set and the problem is solved again. If no
translation fits, text-bearing units in conflict shrink a step (never below readable floors)
and the solve repeats.
"""
import copy
import math

import numpy as np
from scipy.optimize import minimize

from ..blocks import FLOORS
from ..canonical import rotated_extent, union_all

ALLOWANCE = 0.02  # collision check tolerates +0.025 overlap fraction; stay inside it
GRAPHIC_MIN = 0.55  # graphics give up space first, down to this share of their predicted size
MARGIN = 0.04  # safe margin for copy, share of the short side (5th percentile of Crello designers' text: 0.038)
DECOR_MIN = 0.01  # decorations at least this share of the canvas take part in overlap removal
COPY_MIN = 0.3  # then all units in conflict; the engine vetoes compositions that collapse
SWITCH_PENALTY = 1.5  # prefer the source's separating axis unless the other is clearly cheaper


def margin(box, W, H, tW, tH):
    """Safe edge margin for copy with source box [x, y, w, h]: MARGIN of the target's short side,
    or less when the source itself placed the copy nearer the edge (an intentional bleed)."""
    gap = min(box[0], box[1], W - box[0] - box[2], H - box[1] - box[3])
    return min(MARGIN * min(tW, tH), max(0.0, gap) / min(W, H) * min(tW, tH))


def merge(groups):
    """Union overlapping id groups (lists), keeping first-seen order."""
    out = []
    for g in groups:
        g = list(dict.fromkeys(g))
        hit = [o for o in out if set(o) & set(g)]
        for o in hit:
            out.remove(o)
        out.append(list(dict.fromkeys([i for o in hit for i in o] + g)))
    return out


def _area(e):
    return max(float(e['width']), 0.0) * max(float(e['height']), 0.0)


def _inside(e, b):
    ix = max(0.0, min(e['x'] + e['width'], b['x'] + b['width']) - max(e['x'], b['x']))
    iy = max(0.0, min(e['y'] + e['height'], b['y'] + b['height']) - max(e['y'], b['y']))
    return ix * iy / max(_area(e), 1e-6)


def backed(source, loose=False):
    """Groups of a backing (badge, bar, panel) and what sits on it in the source.

    A backing is a graphic drawn beneath, at most a quarter of the canvas. Copy mostly inside it
    and filling a fifth of it or more (a label on a badge or bar, not a line on a large photo), and
    small graphics fully on it, belong to it: moving them apart breaks the design. Each element
    attaches to its smallest backing. Looser definitions (half-canvas panels, any text share) bound
    too much and lost designer agreement on held-out designs, so `loose` (a backing up to three
    quarters of the canvas, copy of any size on it) is only offered as a fallback candidate for
    copy that would otherwise lose its support or contrast.
    """
    max_area, text_share = (0.75, 0.0) if loose else (0.25, 0.2)
    W, H = float(source['canvas']['width']), float(source['canvas']['height'])
    els = source['elements']
    pairs = []
    for e in els:
        best = None
        for b in els:
            if b is e or b['type'] == 'text' or b.get('z_index', 0) >= e.get('z_index', 0) \
                    or _area(b) > max_area * W * H or b.get('rotation') or e.get('rotation'):
                continue
            if e['type'] == 'text':
                ok = _area(b) >= _area(e) and _inside(e, b) >= 0.75 and _area(e) >= text_share * _area(b)
            else:
                # Specks (dots, sparkles) stay free so overlap removal can nudge them aside.
                ok = _area(e) <= 0.5 * _area(b) and _inside(e, b) >= 0.9 and _area(e) >= 0.002 * W * H
            if ok and (best is None or _area(b) < _area(best)):
                best = b
        if best is not None:
            pairs.append([best['id'], e['id']])
    return merge(pairs)


def _boxes(units, design):
    by = {e['id']: e for e in design['elements']}
    out = []
    for u in units:
        ids = [i for i in u if i in by]
        if ids:
            r = union_all(rotated_extent(by[i]) for i in ids)
            out.append((ids, [r.x, r.y, r.w, r.h]))
    return out


def _frac(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, min(a[2] * a[3], b[2] * b[3]))


def _relation(sa, sb, ta, tb, before):
    """(axis, a_first, allowed overlap length) for keeping pair a, b apart.

    Axis 1 = vertical. Disjoint source pairs keep their order with a gap; overlapping source pairs
    may keep up to their source overlap fraction.
    """
    gx = max(sb[0] - (sa[0] + sa[2]), sa[0] - (sb[0] + sb[2]))
    gy = max(sb[1] - (sa[1] + sa[3]), sa[1] - (sb[1] + sb[3]))
    if gx > 0 or gy > 0:
        # Disjoint in the source: keep their order (above or left reads first). A reflow to a very
        # different aspect turns a stack into a row, so the other axis is used when it is clearly
        # cheaper to restore; the source's separating axis alone can be infeasible.
        sa_axis = 1 if gy > gx else 0
        s = (sa[sa_axis] + sa[sa_axis + 2] / 2) <= (sb[sa_axis] + sb[sa_axis + 2] / 2)
        f, g = (ta, tb) if s else (tb, ta)
        cost = [max(0.0, f[k] + f[k + 2] - g[k]) / max(1.0, f[k + 2] + g[k + 2]) for k in (0, 1)]
        cost[1 - sa_axis] *= SWITCH_PENALTY
        return min((0, 1), key=lambda k: (cost[k], k != sa_axis)), s, None
    # Intended overlap in the source: limit it along the axis where it is shallowest now.
    ox = min(ta[0] + ta[2], tb[0] + tb[2]) - max(ta[0], tb[0])
    oy = min(ta[1] + ta[3], tb[1] + tb[3]) - max(ta[1], tb[1])
    axis = 1 if oy / max(1, min(ta[3], tb[3])) <= ox / max(1, min(ta[2], tb[2])) else 0
    s = (sa[axis] + sa[axis + 2] / 2) <= (sb[axis] + sb[axis + 2] / 2)
    other = 1 - axis
    o = max(1.0, min(ta[other] + ta[other + 2], tb[other] + tb[other + 2]) - max(ta[other], tb[other]))
    allowed = (before + ALLOWANCE) * min(ta[2] * ta[3], tb[2] * tb[3]) / o
    return axis, s, allowed


def _solve(units, cons, W, H, gap, margins):
    """Smallest area-weighted unit translations meeting block separations and the canvas.

    cons: (unit of a, unit of b, box of a, box of b, axis, a_first, allowed overlap length).
    """
    n = len(units)
    if not cons:
        return np.zeros((n, 2)), True
    area = np.array([max(1.0, b[2] * b[3]) for _, b in units])
    wgt = np.repeat(area / area.mean(), 2)

    def f(v):
        return float((wgt * v * v).sum())

    def g(v):
        return 2 * wgt * v

    A_rows, lb = [], []
    for i, j, bi, bj, axis, a_first, allowed in cons:
        first, second = (i, j) if a_first else (j, i)
        bf, bs = (bi, bj) if a_first else (bj, bi)
        # second.pos + d_second - (first.pos + first.size + d_first) >= gap  (or >= -allowed)
        need = (-allowed if allowed is not None else gap) - (bs[axis] - bf[axis] - bf[axis + 2])
        row = np.zeros(2 * n); row[2 * second + axis] = 1; row[2 * first + axis] = -1
        A_rows.append(row); lb.append(need)
    for k, (_, b) in enumerate(units):
        for axis, size in ((0, W), (1, H)):
            row = np.zeros(2 * n); row[2 * k + axis] = 1
            m = margins[k]  # stay inside the canvas, copy inside its safe margin
            A_rows.append(row); lb.append(min(0.0, m - b[axis]))
            A_rows.append(-row); lb.append(min(0.0, b[axis] + b[axis + 2] - (size - m)))
    Am, lbv = np.array(A_rows), np.array(lb)
    res = minimize(f, np.zeros(2 * n), jac=g, method='SLSQP',
                   constraints=[{'type': 'ineq', 'fun': lambda v: Am @ v - lbv, 'jac': lambda v: Am}],
                   options={'maxiter': 200, 'ftol': 1e-6})
    ok = res.success and (Am @ res.x - lbv).min() > -1.0
    return res.x.reshape(n, 2), ok


def _move(design, units, d):
    by = {e['id']: e for e in design['elements']}
    for (ids, b), (dx, dy) in zip(units, d):
        for i in ids:
            by[i]['x'] = round(by[i]['x'] + float(dx), 2)
            by[i]['y'] = round(by[i]['y'] + float(dy), 2)


def _shrink(design, A, ids, k):
    """Scale a unit about its centre by k (text keeps its readable floor). False if it cannot."""
    by = {e['id']: e for e in design['elements']}
    for i in ids:
        e = by[i]
        if e['type'] == 'text' and int(e['font_size'] * k) < FLOORS.get(A.roles.get(i, 'body'), 8):
            return False
    r = union_all(rotated_extent(by[i]) for i in ids)
    cx, cy = r.x + r.w / 2, r.y + r.h / 2
    for i in ids:
        e = by[i]
        e.update(x=round(cx + (e['x'] - cx) * k, 2), y=round(cy + (e['y'] - cy) * k, 2),
                 width=round(e['width'] * k, 2), height=round(e['height'] * k, 2))
        if e['type'] == 'text':
            e['font_size'] = max(1, int(e['font_size'] * k))
    return True


def deoverlap(source, design, A, rounds=6, shrink_steps=10, loose=False):
    """Return a copy of design with new block collisions removed, keeping source order.

    Collisions are measured per analysis block, as the engine's check does; blocks that belong
    together (a backing and its copy) translate as one unit.
    """
    design = copy.deepcopy(design)
    W, H = float(design['canvas']['width']), float(design['canvas']['height'])
    gap = 0.012 * min(W, H)
    ids = {e['id'] for e in source['elements']}
    blocks = [b for b in ([i for i in blk.ids if i in ids] for blk in A.all_blocks()) if b]
    # Large graphics the analysis treats as decoration (a panel, a photo strip) also keep their
    # source overlaps: otherwise a panel can slide over the photo it sat beside.
    W0, H0 = float(source['canvas']['width']), float(source['canvas']['height'])
    seen = {i for b in blocks for i in b}
    blocks += [[e['id']] for e in source['elements'] if e['id'] not in seen and e['type'] != 'text'
               and DECOR_MIN * W0 * H0 <= _area(e) < 0.9 * W0 * H0 and not e.get('rotation')]
    groups = merge(blocks + backed(source, loose))
    owner = [next(u for u, grp in enumerate(groups) if set(b) <= set(grp)) for b in blocks]
    src = [b for _, b in _boxes(blocks, source)]
    src_groups = _boxes(groups, source)
    sW, sH = float(source['canvas']['width']), float(source['canvas']['height'])
    by0 = {e['id']: e for e in source['elements']}
    before = {(k, l): _frac(src[k], src[l]) for k in range(len(blocks)) for l in range(k + 1, len(blocks))
              if owner[k] != owner[l]}
    scale = {}
    for _ in range(shrink_steps + 1):
        units = _boxes(groups, design)
        cur = [b for _, b in _boxes(blocks, design)]
        margins = [margin(sg, sW, sH, W, H) if any(by0[i]['type'] == 'text' for i in ids) else 0.0
                   for (ids, _), (_, sg) in zip(units, src_groups)]
        cons, active, ok = [], set(), True
        d = np.zeros((len(units), 2))
        for _ in range(rounds):
            moved = [[b[0] + d[owner[k]][0], b[1] + d[owner[k]][1], b[2], b[3]] for k, b in enumerate(cur)]
            new = [(k, l) for (k, l), f0 in before.items()
                   if (k, l) not in active and _frac(moved[k], moved[l]) > f0 + ALLOWANCE]
            if not new:
                break
            for k, l in new:
                active.add((k, l))
                cons.append((owner[k], owner[l], cur[k], cur[l], *_relation(src[k], src[l], cur[k], cur[l], before[k, l])))
            d, ok = _solve(units, cons, W, H, gap, margins)
            if not ok:
                break
        if ok:
            _move(design, units, d)
            return design
        # No translation fits: shrink the graphics in conflict (to GRAPHIC_MIN of their size),
        # then everything in conflict together (to COPY_MIN; copy never below its readable floor),
        # and try again. Past that the overlap stays and the engine's collision check vetoes it.
        by = {e['id']: e for e in design['elements']}
        involved = [units[u][0] for u in sorted({u for c in cons for u in c[:2]})]
        room = lambda ids, floor: scale.get(tuple(ids), 1.0) * 0.9 >= floor
        has_text = lambda ids: any(by[i]['type'] == 'text' for i in ids)
        graphics = [ids for ids in involved if not has_text(ids) and room(ids, GRAPHIC_MIN)]
        rest = [ids for ids in involved if room(ids, COPY_MIN)]
        shrunk = False
        for ids in graphics or rest:
            if _shrink(design, A, ids, 0.9):
                shrunk = True
                scale[tuple(ids)] = scale.get(tuple(ids), 1.0) * 0.9
        if not shrunk:
            break
    return design
