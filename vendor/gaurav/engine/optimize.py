"""Local search over a candidate layout, down the energy of engine/energy.py.

The variables are each content unit's position and uniform scale (a group
moves and scales as one piece, images keep their aspect, fonts follow the
scale and are re-fitted to their boxes); a single text may also re-wrap
narrower or wider. Moves, per unit and in this order:

  snap      slide the unit so one of its edges / centres meets the nearest
            line of another unit (within SNAP_RANGE of the canvas)
  nudge     translate by +-NUDGE of the canvas on either axis
  scale     grow / shrink by SCALE_STEP about the unit's anchored point
  rewrap    (single texts) box REWRAP_NARROW / REWRAP_WIDE as wide, height
            from the measurer, font kept; the text's own alignment decides
            which edge stays put
  swap      exchange places with the next unit of its stack or row (never
            two text-bearing units: text keeps the designer's sequence)

Greedy best-improvement: for every unit (most important first) each move is
tried in turn, the one that lowers the energy most is kept, everything else
is undone; sweeps repeat while something improved. The search is
deterministic: no randomness, fixed iteration order, and the budget is a
number of energy evaluations (`max_evals`), not a clock; `time_cap_s` is a
safety net for a pathological case only (a run that hits it reports it in
`Result.capped`). A move is accepted only if it introduces no hard fail
the seed did not have (`energy.new_hard`), so the optimiser never turns a
clean layout into a defective one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from engine.analyze import Analysis
from engine.energy import Energy, Layout, new_hard
from engine.geometry import Box
from engine.rectify import Live
from engine.relations import LINES, X_LINES, line_value
from engine.textfit import TextMeasurer

DEFAULT_MAX_EVALS = 160     # energy evaluations per candidate (the deterministic budget)
MAX_SWEEPS = 3
TIME_CAP_S = 1.5            # safety net only; see the module docstring
EPS = 0.1                   # a move must lower the energy by more than this (energy units; the soft part spans 0-100):
                            # a negligible gain is not worth touching a design and would make the search path-sensitive
SNAP_RANGE = 0.06           # snap to a line at most this x the canvas span away
NUDGE = 0.02                # translate by this x the canvas span
SCALE_STEP = 0.1
REWRAP_NARROW, REWRAP_WIDE = 0.75, 1.3
MIN_UNITS = 2               # nothing to arrange below this


@dataclass
class Result:
    elements: list[dict[str, Any]]
    before: Energy
    after: Energy
    evals: int = 0
    accepted: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    capped: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.accepted)

    def summary(self) -> dict[str, Any]:
        return {"before": round(self.before.total, 3), "after": round(self.after.total, 3),
                "evals": self.evals, "moves": list(self.accepted), "elapsed": round(self.elapsed, 3),
                "capped": self.capped}


Move = tuple[str, list[Live], Callable[[], None]]


def optimize(analysis: Analysis, source: dict[str, Any], elements: list[dict[str, Any]], target: Box, *,
             max_evals: int = DEFAULT_MAX_EVALS, max_sweeps: int = MAX_SWEEPS, time_cap_s: float = TIME_CAP_S,
             measurer: TextMeasurer | None = None, weights: dict[str, float] | None = None) -> Result:
    """Return the optimised copy of `elements` (same ids, same order)."""
    t0 = time.time()
    lay = Layout(analysis, source, elements, target, measurer)
    seed = lay.evaluate(weights)
    cur = seed
    accepted: list[str] = []
    capped = False
    if len(lay.lives) < MIN_UNITS:
        return Result(lay.elements, seed, seed, lay.evals, accepted, time.time() - t0)
    order = sorted(lay.lives, key=lambda l: (-l.unit.importance, l.unit.reading_index, l.unit.key))
    for _sweep in range(max(1, max_sweeps)):
        improved = False
        for l in order:
            if lay.evals >= max_evals:
                break
            if time.time() - t0 > time_cap_s:
                capped = True
                break
            best: tuple[float, str, list[tuple[Live, tuple]]] | None = None
            for name, involved, apply in proposals(lay, l):
                if lay.evals >= max_evals:
                    break
                snaps = [(x, lay.snapshot(x)) for x in involved]
                try:
                    apply()
                    e = lay.evaluate(weights)
                    ok = e.total < cur.total - EPS and not new_hard(seed, e)
                    if ok and (best is None or e.total < best[0]):
                        best = (e.total, name, [(x, lay.snapshot(x)) for x in involved])
                        best_energy = e
                finally:
                    for x, s in snaps:
                        lay.restore(x, s)
            if best is not None:
                for x, s in best[2]:
                    lay.restore(x, s)
                cur = best_energy
                accepted.append(f"{l.unit.key}: {best[1]}")
                improved = True
        if not improved or lay.evals >= max_evals or capped:
            break
    return Result(lay.elements, seed, cur, lay.evals, accepted, time.time() - t0, capped)


# ---------------------------------------------------------------------------
# proposals


def proposals(lay: Layout, l: Live) -> list[Move]:
    out: list[Move] = []
    out += snap_moves(lay, l)
    out += nudge_moves(lay, l)
    out += scale_moves(lay, l)
    out += rewrap_moves(lay, l)
    out += swap_moves(lay, l)
    return out


def _translate(lay: Layout, l: Live, dx: float, dy: float) -> Callable[[], None]:
    return lambda: lay.move(l, dx, dy)


def snap_moves(lay: Layout, l: Live) -> list[Move]:
    """For each line type the nearest other unit's line within SNAP_RANGE
    (and farther than half the alignment tolerance: already aligned lines
    are not re-snapped)."""
    vb = lay.vis(l)
    tol = lay.src_rel.tol
    out: list[Move] = []
    seen: set[tuple[str, float]] = set()
    for line in LINES:
        axis = "x" if line in X_LINES else "y"
        span = lay.target.w if axis == "x" else lay.target.h
        mine = line_value(vb, line)
        best: tuple[float, str] | None = None
        for o in lay.lives:
            if o is l:
                continue
            d = line_value(lay.vis(o), line) - mine
            if 0.5 * tol < abs(d) <= SNAP_RANGE * span and (best is None or abs(d) < abs(best[0])):
                best = (d, o.unit.key)
        if best is None:
            continue
        key = (axis, round(best[0], 1))
        if key in seen:
            continue
        seen.add(key)
        dx, dy = (best[0], 0.0) if axis == "x" else (0.0, best[0])
        out.append((f"snap {line} to {best[1]}", [l], _translate(lay, l, dx, dy)))
    return out


def nudge_moves(lay: Layout, l: Live) -> list[Move]:
    out: list[Move] = []
    for dx, dy, name in ((NUDGE * lay.target.w, 0.0, "nudge right"), (-NUDGE * lay.target.w, 0.0, "nudge left"),
                         (0.0, NUDGE * lay.target.h, "nudge down"), (0.0, -NUDGE * lay.target.h, "nudge up")):
        out.append((name, [l], _translate(lay, l, dx, dy)))
    return out


def _anchor_point(lay: Layout, l: Live) -> tuple[float, float]:
    fb = lay.frame(l)
    u = l.unit
    ax = fb.x if u.anchor_x == "left" else fb.right if u.anchor_x == "right" else fb.cx
    ay = fb.y if u.anchor_y == "top" else fb.bottom if u.anchor_y == "bottom" else fb.cy
    return ax, ay


def _font_exact_factor(lay: Layout, l: Live, f: float) -> float:
    """Snap a scale factor so the unit's largest font lands on an integer:
    fonts are drawn at int(font_size), and a box scaled by f while its
    font rounds down would let an attached bar drift from its heading."""
    fonts = [v for v in l.fonts.values() if v >= 1.0]
    if not fonts:
        return f
    big = max(fonts)
    target = max(1.0, float(int(big * f)))
    return target / big


def scale_moves(lay: Layout, l: Live) -> list[Move]:
    out: list[Move] = []
    for f, name in ((1.0 + SCALE_STEP, "grow"), (1.0 - SCALE_STEP, "shrink")):
        f = _font_exact_factor(lay, l, f)
        if abs(f - 1.0) < 1e-6:
            continue

        def apply(f=f):
            ax, ay = _anchor_point(lay, l)
            lay.scale(l, f, ax, ay)
        out.append((f"{name} x{f:.2f}", [l], apply))
    return out


def rewrap_moves(lay: Layout, l: Live) -> list[Move]:
    """A single, unrotated text may take a narrower or wider box; the
    height follows the wrapped lines and the text's own alignment decides
    which edge stays put. Skipped when the new width cannot hold the text."""
    if len(l.boxes) != 1:
        return []
    (eid, b), = l.boxes.items()
    el = lay.by_id[eid]
    if el["type"] != "text" or float(el.get("rotation", 0.0) or 0.0) or not str(el.get("content", "")).strip():
        return []
    font = l.fonts.get(eid, float(el["font_size"]))
    out: list[Move] = []
    for k, name in ((REWRAP_NARROW, "rewrap narrower"), (REWRAP_WIDE, "rewrap wider")):
        w = b.w * k
        if w < 4.0 or w > lay.target.w:
            continue
        if not lay.m.fits(el, font, w, 1e9):
            continue
        h = lay.m.natural_height(el, font, w)

        def apply(w=w, h=h):
            align = str(el.get("text_align", "left") or "left")
            dx = (b.w - w) / 2.0 if align == "center" else (b.w - w) if align == "right" else 0.0
            l.boxes[eid] = Box(b.x + dx, b.y, w, max(h, 1.0))
            lay.r.invalidate(l)
            lay._sync(l)
        out.append((name, [l], apply))
    return out


def _has_text(lay: Layout, l: Live) -> bool:
    return any(lay.by_id[e]["type"] == "text" for e in l.boxes)


def swap_moves(lay: Layout, l: Live) -> list[Move]:
    """Exchange places with the unit that follows `l` in its stack (y) or
    row (x); the gap between them is kept. Never between two text-bearing
    units: text reads in the designer's sequence (a headline over its
    subhead is a lockup, not two interchangeable blocks); a picture may
    change sides with a text column."""
    out: list[Move] = []
    mine_text = _has_text(lay, l)
    for axis in ("y", "x"):
        for chain in lay.r._chains(axis):
            keys = [c.unit.key for c in chain]
            if l.unit.key not in keys:
                continue
            i = keys.index(l.unit.key)
            if i + 1 >= len(chain):
                continue
            o = chain[i + 1]
            if mine_text and _has_text(lay, o):
                continue
            va, vb = lay.vis(l), lay.vis(o)

            def apply(o=o, va=va, vb=vb, axis=axis):
                if axis == "y":
                    g = vb.y - va.bottom
                    lay.move(o, 0.0, va.y - vb.y)
                    lay.move(l, 0.0, vb.h + g)
                else:
                    g = vb.x - va.right
                    lay.move(o, va.x - vb.x, 0.0)
                    lay.move(l, vb.w + g, 0.0)
            out.append((f"swap with {o.unit.key} ({axis})", [l, o], apply))
    return out


__all__ = ["optimize", "Result", "proposals", "DEFAULT_MAX_EVALS"]
