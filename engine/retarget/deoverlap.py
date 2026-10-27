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


def _boxes(A, design):
    by = {e['id']: e for e in design['elements']}
    out = []
    for b in A.all_blocks():
        ids = [i for i in b.ids if i in by]
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

    Axis 1 = vertical. Disjoint source pairs keep their separating order with a gap (negative
    allowed overlap); overlapping source pairs may keep up to their source overlap fraction.
    """
    gx = max(sb[0] - (sa[0] + sa[2]), sa[0] - (sb[0] + sb[2]))
    gy = max(sb[1] - (sa[1] + sa[3]), sa[1] - (sb[1] + sb[3]))
    if gx > 0 and gy > 0:
        # Separated diagonally: keep whichever order costs less to restore now.
        need_x = min(ta[0] + ta[2] - tb[0], tb[0] + tb[2] - ta[0])
        need_y = min(ta[1] + ta[3] - tb[1], tb[1] + tb[3] - ta[1])
        axis = 1 if need_y / max(1, ta[3] + tb[3]) <= need_x / max(1, ta[2] + tb[2]) else 0
    elif gy > 0 or gx > 0:
        axis = 1 if gy > gx else 0
    else:
        # Intended overlap in the source: limit it along the axis where it is shallowest now.
        ox = min(ta[0] + ta[2], tb[0] + tb[2]) - max(ta[0], tb[0])
        oy = min(ta[1] + ta[3], tb[1] + tb[3]) - max(ta[1], tb[1])
        axis = 1 if oy / max(1, min(ta[3], tb[3])) <= ox / max(1, min(ta[2], tb[2])) else 0
    s = (sa[axis] + sa[axis + 2] / 2) <= (sb[axis] + sb[axis + 2] / 2)
    if gx > 0 or gy > 0:
        return axis, s, None
    other = 1 - axis
    o = max(1.0, min(ta[other] + ta[other + 2], tb[other] + tb[other + 2]) - max(ta[other], tb[other]))
    allowed = (before + ALLOWANCE) * min(ta[2] * ta[3], tb[2] * tb[3]) / o
    return axis, s, allowed


def _solve(units, src, cons, W, H, gap):
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
    for i, j, axis, a_first, allowed in cons:
        bi, bj = units[i][1], units[j][1]
        first, second = (i, j) if a_first else (j, i)
        bf, bs = (bi, bj) if a_first else (bj, bi)
        # second.pos + d_second - (first.pos + first.size + d_first) >= gap  (or >= -allowed)
        need = (-allowed if allowed is not None else gap) - (bs[axis] - bf[axis] - bf[axis + 2])
        row = np.zeros(2 * n); row[2 * second + axis] = 1; row[2 * first + axis] = -1
        A_rows.append(row); lb.append(need)
    for k, (_, b) in enumerate(units):
        for axis, size in ((0, W), (1, H)):
            row = np.zeros(2 * n); row[2 * k + axis] = 1
            A_rows.append(row); lb.append(min(0.0, -b[axis]))           # stay right/below 0
            A_rows.append(-row); lb.append(min(0.0, b[axis] + b[axis + 2] - size))  # and inside
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


def deoverlap(source, design, A, rounds=6, shrink_steps=5):
    """Return a copy of design with new unit collisions removed, keeping source order."""
    design = copy.deepcopy(design)
    W, H = float(design['canvas']['width']), float(design['canvas']['height'])
    gap = 0.012 * min(W, H)
    src = {tuple(ids): b for ids, b in _boxes(A, source)}
    for _ in range(shrink_steps + 1):
        units = [(ids, b) for ids, b in _boxes(A, design) if tuple(ids) in src]
        before = {}
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                before[i, j] = _frac(src[tuple(units[i][0])], src[tuple(units[j][0])])
        cons, active, ok = [], set(), True
        d = np.zeros((len(units), 2))
        for _ in range(rounds):
            moved = [(ids, [b[0] + dx, b[1] + dy, b[2], b[3]]) for (ids, b), (dx, dy) in zip(units, d)]
            new = [(i, j) for (i, j), f0 in before.items()
                   if (i, j) not in active and _frac(moved[i][1], moved[j][1]) > f0 + ALLOWANCE]
            if not new:
                break
            for i, j in new:
                active.add((i, j))
                cons.append((i, j, *_relation(src[tuple(units[i][0])], src[tuple(units[j][0])],
                                               units[i][1], units[j][1], before[i, j])))
            d, ok = _solve(units, src, cons, W, H, gap)
            if not ok:
                break
        if ok:
            _move(design, units, d)
            return design
        # No translation fits: shrink the text-bearing units in conflict and try again.
        by = {e['id']: e for e in design['elements']}
        involved = {k for c in cons for k in c[:2]}
        shrunk = False
        for k in involved:
            if any(by[i]['type'] == 'text' for i in units[k][0]):
                shrunk |= _shrink(design, A, units[k][0], 0.9)
        if not shrunk:
            break
    return design
