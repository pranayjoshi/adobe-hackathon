"""Joint minimal-move constraint pass (v11, engine/flags SOLVE).

The rectifier fixes defects greedily, pair by pair, and its snap / even-gap
passes accept a move only when nothing else objects. This pass instead
solves the positions of all content units at once, per axis, as the
smallest weighted displacement (important units are stiffer) that satisfies

  * alignment   edges / centres that are nearly on one line (within
                ALIGN_FRAC of the short side) land exactly on it
  * rhythm      a stack (row) whose gaps are nearly even (max / min <=
                EVEN_RATIO) gets exactly even gaps
  * order       two units that do not overlap on this axis but share the
                other keep their order and at least min(their gap, the
                rectifier's gap) between them, so nothing new collides
  * layering    units that overlap on both axes move together
  * canvas      content stays inside the rectifier's margin (or no further
                out than it already was)

Units only translate (fonts and sizes are untouched, so no new overflow);
the problem is a small convex QP solved with Dykstra's alternating
projections in the weighted metric (numpy, no native code). The result is
offered to the judge as a candidate "<seed> +solve" (engine/resize.py), the
way the optimiser's "+opt" layouts are, so it is kept only where it scores.
"""

from __future__ import annotations

import copy
from typing import Any

from engine.analyze import Analysis
from engine.geometry import Box, clamp
from engine.rectify import EPS, GAP_FRAC, MARGIN_FRAC, Live, _lives, unit_visible_box
from engine.textfit import TextMeasurer, default_measurer

ALIGN_FRAC = 0.012          # edges / centres this close (share of the short side) are meant to be one line
EVEN_RATIO = 1.6            # stack gaps this even (max / min) are meant to be equal
MAX_SWEEPS = 300
TOL = 0.05                  # px: stop when no coordinate moved more than this in a sweep
MAX_MOVE_FRAC = 0.04        # a solution that moves any unit more than this share of the canvas is dropped


def solve(analysis: Analysis, elements: list[dict[str, Any]], target: Box,
          measurer: TextMeasurer | None = None) -> list[dict[str, Any]] | None:
    """A copy of `elements` with the content units translated to the
    jointly solved positions, or None when nothing changes (or the solver
    would have to move something further than MAX_MOVE_FRAC)."""
    m = measurer or default_measurer()
    out = copy.deepcopy(elements)
    by_id = {el["id"]: el for el in out}
    lives = _lives(analysis, by_id)
    if len(lives) < 2:
        return None
    vis = [unit_visible_box(l, by_id, m) for l in lives]
    frames = [l.frame(by_id) for l in lives]
    mn = min(target.w, target.h)
    margin = clamp(MARGIN_FRAC * mn, 1.0, 16.0)
    gap = clamp(GAP_FRAC * mn, 2.0, 12.0)
    tol = ALIGN_FRAC * mn
    weights = [1.0 + 4.0 * l.importance for l in lives]
    moves = []
    for axis in ("x", "y"):
        t = _solve_axis(axis, lives, vis, frames, weights, target, margin, gap, tol)
        if t is None:
            return None
        moves.append(t)
    tx, ty = moves
    if all(abs(a) < 0.5 for a in tx) and all(abs(b) < 0.5 for b in ty):
        return None
    limit = MAX_MOVE_FRAC * max(target.w, target.h)
    if any(abs(a) > limit for a in tx) or any(abs(b) > limit for b in ty):
        return None
    for l, dx, dy in zip(lives, tx, ty):
        for eid in l.boxes:
            el = by_id[eid]
            el["x"] = float(el["x"]) + dx
            el["y"] = float(el["y"]) + dy
    return out


def _lo(b: Box, axis: str) -> float:
    return b.x if axis == "x" else b.y


def _hi(b: Box, axis: str) -> float:
    return b.right if axis == "x" else b.bottom


def _mid(b: Box, axis: str) -> float:
    return b.cx if axis == "x" else b.cy


def _cross_overlap(a: Box, b: Box, axis: str) -> float:
    """Overlap on the *other* axis (shared extent)."""
    if axis == "x":
        return min(a.bottom, b.bottom) - max(a.y, b.y)
    return min(a.right, b.right) - max(a.x, b.x)


def _solve_axis(axis: str, lives: list[Live], vis: list[Box], frames: list[Box], weights: list[float],
                target: Box, margin: float, gap: float, tol: float) -> list[float] | None:
    n = len(lives)
    # constraints as (coeffs {i: c}, rhs, kind): sum c_i t_i  (= | <=)  rhs
    eqs: list[tuple[dict[int, float], float]] = []       # layering
    align: list[tuple[dict[int, float], float]] = []
    rhythm: list[tuple[dict[int, float], float]] = []
    ineqs: list[tuple[dict[int, float], float]] = []

    # layering: units overlapping on both axes move together
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vis[i], vis[j]
            if _hi(a, axis) > _lo(b, axis) + EPS and _hi(b, axis) > _lo(a, axis) + EPS \
                    and _cross_overlap(a, b, axis) > EPS:
                eqs.append(({i: 1.0, j: -1.0}, 0.0))

    # alignment: the closest near-miss edge pair per unit pair
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vis[i], vis[j]
            best = None
            for f in (_lo, _mid, _hi):
                d = f(b, axis) - f(a, axis)
                if EPS < abs(d) <= tol and (best is None or abs(d) < abs(best)):
                    best = d
            if best is not None:
                # f(a) + t_i = f(b) + t_j  ->  t_i - t_j = d
                align.append(({i: 1.0, j: -1.0}, best))

    # order and minimum gap between units that share the other axis
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = vis[i], vis[j]
            g = _lo(b, axis) - _hi(a, axis)
            if g < -EPS or _cross_overlap(a, b, axis) <= EPS:
                continue
            need = min(g, gap)
            # lo(b) + t_j - hi(a) - t_i >= need  ->  t_i - t_j <= g - need
            ineqs.append(({i: 1.0, j: -1.0}, g - need))

    # rhythm: chains along this axis with nearly even gaps
    for chain in _chains(vis, axis):
        if len(chain) < 3:
            continue
        gaps = [_lo(vis[chain[k + 1]], axis) - _hi(vis[chain[k]], axis) for k in range(len(chain) - 1)]
        lo_g, hi_g = min(gaps), max(gaps)
        if lo_g < 0.0 or hi_g - lo_g <= 0.5 or hi_g > EVEN_RATIO * max(lo_g, 1.0):
            continue
        for k in range(len(gaps) - 1):
            p, q, r = chain[k], chain[k + 1], chain[k + 2]
            # (lo_r + t_r - hi_q - t_q) - (lo_q + t_q - hi_p - t_p) = 0
            coeffs: dict[int, float] = {}
            for idx, c in ((r, 1.0), (q, -2.0), (p, 1.0)):
                coeffs[idx] = coeffs.get(idx, 0.0) + c
            rhythm.append((coeffs, gaps[k] - gaps[k + 1]))

    # canvas: no further out than the margin (or than it already was)
    size = target.w if axis == "x" else target.h
    for i in range(n):
        f = frames[i]
        lo_bound = min(0.0, margin - _lo(f, axis))         # t_i >= lo_bound
        hi_bound = max(0.0, size - margin - _hi(f, axis))  # t_i <= hi_bound
        ineqs.append(({i: -1.0}, -lo_bound))
        ineqs.append(({i: 1.0}, hi_bound))

    if not align and not rhythm:
        return [0.0] * n
    # alignments and rhythm can contradict each other (or the order): the
    # whole set first, then each family alone, else nothing moves
    for subset in (align + rhythm, align, rhythm):
        if not subset:
            continue
        t = _dykstra(n, weights, eqs + subset, ineqs)
        if t is not None:
            return t
    return [0.0] * n


def _chains(vis: list[Box], axis: str) -> list[list[int]]:
    """Runs of units following each other along `axis` and sharing most of
    the other axis (the rectifier's even-gap chains, on indices)."""
    order = sorted(range(len(vis)), key=lambda i: _lo(vis[i], axis))
    nxt: dict[int, int] = {}
    taken: set[int] = set()
    for pos, i in enumerate(order):
        best = None
        for j in order[pos + 1:]:
            g = _lo(vis[j], axis) - _hi(vis[i], axis)
            span = min(vis[i].w, vis[j].w) if axis == "y" else min(vis[i].h, vis[j].h)
            if g < -0.5 or _cross_overlap(vis[i], vis[j], axis) < 0.5 * span:
                continue
            if best is None or g < best[0]:
                best = (g, j)
        if best is not None and best[1] not in taken:
            nxt[i] = best[1]
            taken.add(best[1])
    chains = []
    for i in order:
        if i in taken:
            continue
        c = [i]
        while c[-1] in nxt:
            c.append(nxt[c[-1]])
        if len(c) >= 2:
            chains.append(c)
    return chains


def _dykstra(n: int, weights: list[float], eqs: list[tuple[dict[int, float], float]],
             ineqs: list[tuple[dict[int, float], float]]) -> list[float] | None:
    """Closest point to 0 (weighted norm sum w_i t_i^2) in the intersection
    of the hyperplanes `eqs` and half-spaces `ineqs`, by Dykstra's algorithm.
    None when the constraints are inconsistent (no convergence)."""
    import numpy as np

    winv = 1.0 / np.asarray(weights, dtype=float)
    cons = []
    for coeffs, rhs in eqs:
        a = np.zeros(n)
        for i, c in coeffs.items():
            a[i] += c
        cons.append((a, rhs, True))
    for coeffs, rhs in ineqs:
        a = np.zeros(n)
        for i, c in coeffs.items():
            a[i] += c
        cons.append((a, rhs, False))
    norms = [float(a @ (winv * a)) for a, _, _ in cons]
    t = np.zeros(n)
    corr = [np.zeros(n) for _ in cons]
    for _ in range(MAX_SWEEPS):
        prev = t.copy()
        for k, (a, rhs, eq) in enumerate(cons):
            if norms[k] <= 1e-12:
                continue
            y = t + corr[k]
            v = float(a @ y) - rhs
            if eq or v > 0.0:
                p = y - winv * a * (v / norms[k])
            else:
                p = y
            corr[k] = y - p
            t = p
        if float(np.max(np.abs(t - prev))) < TOL:
            break
    else:
        return None
    # feasibility check (an inconsistent set converges to a compromise)
    for a, rhs, eq in cons:
        v = float(a @ t) - rhs
        if (eq and abs(v) > 0.75) or (not eq and v > 0.75):
            return None
    return [float(x) for x in t]
