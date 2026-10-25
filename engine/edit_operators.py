"""Typed geometry edits, defect neighbourhoods and the controller expansion (guide 07).

Every operator receives an immutable source, the current assessed candidate,
analysis/group information, exact locks and a shared budget, and produces a
``GeometryEdit``: an allowlisted geometry patch (``x, y, width, height,
font_size`` plus the scoped background ``fit: cover`` exception) with the
affected ids, a reason and an estimated cost. Edits never touch content,
fonts, colours, assets, ids or layer order, so every design they produce
survives ``engine.materialize.apply_edit``.

Published for Codex's controller::

    expand(candidate, budget, *, source, analysis, locks) -> Iterable[(operation, design)]

The legacy hill-climb operators of ``engine/refine.py`` are wrapped unchanged;
compound neighbourhood operators (widen + shift, redistribute row + refit,
rebalance column + restore padding, CP-SAT neighbourhood repair with frozen
unaffected groups) and photo-support operators are added on top.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from .canonical import MUTABLE_FIELDS, Box, element_box, rotated_extent, union_all

EDIT_VERSION = 'edit-operators/1'
GEOMETRY = tuple(sorted(MUTABLE_FIELDS))
BLOCKING = ('overlap.text', 'grouping.rigid', 'legibility.overflow')


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class GeometryEdit:
    """One typed edit. ``changes`` maps element id -> allowlisted fields."""
    operation: str
    family: str
    affected_ids: tuple[str, ...]
    changes: dict[str, dict[str, Any]] = field(repr=False)
    reason: str = ''
    estimated_cost: float = 1.0
    neighborhood: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {'operation': self.operation, 'family': self.family, 'affected_ids': list(self.affected_ids),
                'reason': self.reason, 'estimated_cost': round(self.estimated_cost, 3),
                'neighborhood': list(self.neighborhood), 'changes': self.changes}


class SearchBudget:
    """Deterministic evaluation budget + optional deadline + cancellation token.

    ``clock`` is injectable so tests never depend on wall-clock timing.
    """

    def __init__(self, *, max_evals: int | None = None, deadline: float | None = None,
                 cancelled: Callable[[], bool] | None = None, clock: Callable[[], float] = time.monotonic):
        self.max_evals, self.deadline, self.cancelled, self.clock = max_evals, deadline, cancelled, clock
        self.evals = 0
        self.started = clock()

    def stop_reason(self) -> str | None:
        if self.cancelled and self.cancelled():
            return 'cancelled'
        if self.deadline is not None and self.clock() >= self.deadline:
            return 'deadline'
        if self.max_evals is not None and self.evals >= self.max_evals:
            return 'eval_budget'
        return None

    def remaining(self) -> float:
        return math.inf if self.deadline is None else max(0.0, self.deadline - self.clock())

    def charge(self, n: int = 1) -> None:
        self.evals += n


def _stop(budget) -> str | None:
    """Works with SearchBudget, contracts.Budget or None."""
    if budget is None:
        return None
    try:
        return budget.stop_reason()
    except AttributeError:
        return None


def design_hash(design: dict[str, Any], ndigits: int = 1) -> str:
    """Hash of materialized geometry only (rounded) - dedupes near-identical states."""
    rows = []
    for e in sorted(design['elements'], key=lambda e: str(e['id'])):
        rows.append([str(e['id'])] + [round(float(e.get(k, 0) or 0), ndigits) for k in ('x', 'y', 'width', 'height')]
                    + [int(e.get('font_size', 0) or 0), str(e.get('fit', ''))])
    return hashlib.sha256(json.dumps([design['canvas'], rows]).encode()).hexdigest()[:20]


def diff_edit(before: dict[str, Any], after: dict[str, Any], operation: str, *, family: str | None = None,
              reason: str = '', cost: float = 1.0, neighborhood: Iterable[str] = ()) -> GeometryEdit | None:
    """Typed edit between two designs; ``None`` when nothing changed.

    Raises ``ValueError`` if anything outside the geometry allowlist changed
    (the scoped ``fit`` exception is re-checked by the materializer).
    """
    if before['canvas'] != after['canvas']:
        raise ValueError('edit changes the canvas')
    old = {e['id']: e for e in before['elements']}
    if [e['id'] for e in after['elements']] != [e['id'] for e in before['elements']]:
        raise ValueError('edit changes element coverage or order')
    changes: dict[str, dict[str, Any]] = {}
    for e in after['elements']:
        o = old[e['id']]
        keys = set(o) | set(e)
        bad = [k for k in keys if o.get(k) != e.get(k) and k not in MUTABLE_FIELDS and k != 'fit']
        if bad:
            raise ValueError('edit changes immutable field(s): ' + ', '.join(sorted(bad)))
        delta = {}
        for k in sorted(keys):
            if o.get(k) == e.get(k):
                continue
            v = e.get(k)
            if k in MUTABLE_FIELDS:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError('non-finite geometry')
                if k != 'font_size' and abs(float(v) - float(o.get(k, 0))) < 1e-6:
                    continue
            delta[k] = v
        if delta:
            changes[e['id']] = delta
    if not changes:
        return None
    return GeometryEdit(operation, family or operation, tuple(sorted(changes)), changes, reason, cost, tuple(neighborhood))


def apply(design: dict[str, Any], edit: GeometryEdit) -> dict[str, Any]:
    out = copy.deepcopy(design)
    by = {e['id']: e for e in out['elements']}
    for eid, patch in edit.changes.items():
        by[eid].update(patch)
    return out


# ------------------------------------------------------------------ context
@dataclass
class EditContext:
    source: dict[str, Any]
    analysis: Any
    locks: dict[str, dict[str, float]]
    budget: Any = None
    photo_mode: str = 'heuristic'
    locked: frozenset = frozenset()
    _photo: dict = field(default_factory=dict, repr=False)
    allow_solver: bool = True

    @classmethod
    def create(cls, source, analysis, locks=None, budget=None, design=None, photo_mode=None):
        from .photo_analysis import photo_mode as _pm
        if isinstance(locks, dict):
            rects = {k: dict(v) for k, v in locks.items()}
        else:
            ids = set(locks or ())
            by = {e['id']: e for e in (design or source)['elements']}
            rects = {i: {k: by[i][k] for k in ('x', 'y', 'width', 'height')} for i in ids if i in by}
        return cls(source, analysis, rects, budget, _pm(photo_mode), frozenset(rects))


# ------------------------------------------------------- dependency graph
@dataclass
class DependencyGraph:
    """Blocks linked by collisions, alignment, groups, photo support, adjacency."""
    boxes: dict[str, Box]
    blocks: dict[str, Any]
    owner: dict[str, str]
    edges: dict[str, dict[str, set[str]]]

    def link(self, a: str | None, b: str | None, kind: str) -> None:
        if a is None or b is None or a == b or a not in self.blocks or b not in self.blocks:
            return
        self.edges.setdefault(a, {}).setdefault(b, set()).add(kind)
        self.edges.setdefault(b, {}).setdefault(a, set()).add(kind)

    def neighbours(self, bid: str) -> list[tuple[str, set[str]]]:
        return sorted(self.edges.get(bid, {}).items())

    def neighborhood(self, seeds: Iterable[str], max_size: int = 4, kinds: tuple[str, ...] = ()) -> tuple[str, ...]:
        rank = {'collision': 0, 'overlap': 0, 'support': 1, 'group': 1, 'align': 2, 'adjacent': 3, 'graph': 2}
        order = [s for s in dict.fromkeys(seeds) if s in self.blocks]
        out = list(order)
        frontier = list(order)
        while frontier and len(out) < max_size:
            cand = []
            for b in frontier:
                for n, ks in self.neighbours(b):
                    if n in out or (kinds and not ks.intersection(kinds)):
                        continue
                    cand.append((min(rank.get(k, 4) for k in ks), n))
            if not cand:
                break
            frontier = []
            for _, n in sorted(cand):
                if n not in out and len(out) < max_size:
                    out.append(n); frontier.append(n)
        return tuple(out)


def _source_hosts(source, A):
    from .photo_analysis import source_photo_supports
    return source_photo_supports(source, A)


def build_graph(ctx: EditContext, r: dict[str, Any]) -> DependencyGraph:
    from .quality import units
    A, design = ctx.analysis, r['design']
    cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    u = units(A, design)
    g = DependencyGraph({b.id: box for b, box in u}, {b.id: b for b, _ in u},
                        {i: b.id for b, _ in u for i in b.ids}, {})
    for c in (r.get('diagnostics') or {}).get('collisions', []):
        g.link(c.get('a'), c.get('b'), 'collision')
    t2 = (r.get('rubric') or {}).get('tier2') or {}
    for issue in t2.get('defects', []) + t2.get('warnings', []):
        owners = sorted({g.owner[i] for i in issue.get('ids', []) if i in g.owner})
        kind = 'overlap' if issue['id'].startswith('overlap') else 'group' if issue['id'].startswith('grouping') else 'graph'
        for n, a in enumerate(owners):
            for b in owners[n + 1:]:
                g.link(a, b, kind)
    for rel in getattr(A, 'relationships', []) or []:
        if rel.get('kind') == 'left_alignment':
            g.link(rel.get('a'), rel.get('b'), 'align')
    src = {e['id']: e for e in ctx.source['elements']}
    by_group: dict[str, set[str]] = {}
    for i, bid in g.owner.items():
        gid = src.get(i, {}).get('group_id')
        if gid:
            by_group.setdefault(gid, set()).add(bid)
    for members in by_group.values():
        m = sorted(members)
        for n, a in enumerate(m):
            for b in m[n + 1:]:
                g.link(a, b, 'group')
    hosts = _source_hosts(ctx.source, A)
    by_host: dict[str, set[str]] = {}
    for t, h in hosts.items():
        if t in g.owner:
            by_host.setdefault(h, set()).add(g.owner[t])
    for members in by_host.values():
        m = sorted(members)
        for n, a in enumerate(m):
            for b in m[n + 1:]:
                g.link(a, b, 'support')
    # Optional C1 analysis graph (engine/relations.py); absent -> deterministic fallback above.
    graph = getattr(A, 'graph', None)
    for edge in (getattr(graph, 'edges', None) or []) if graph is not None else []:
        try:
            g.link(g.owner.get(edge['a']), g.owner.get(edge['b']), 'graph' if edge.get('kind') != 'overlaps' else 'overlap')
        except (TypeError, KeyError, AttributeError):
            continue
    near = .03 * min(cw, ch)
    ids = sorted(g.boxes)
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            x, y = g.boxes[a], g.boxes[b]
            dx = max(0.0, max(x.x, y.x) - min(x.x1, y.x1)); dy = max(0.0, max(x.y, y.y) - min(x.y1, y.y1))
            if math.hypot(dx, dy) <= near:
                g.link(a, b, 'adjacent')
    return g


def defect_seeds(r: dict[str, Any], g: DependencyGraph) -> list[tuple[int, str, tuple[str, ...]]]:
    """(priority, code, block ids) for each measured problem; lower priority first."""
    out = []
    t2 = (r.get('rubric') or {}).get('tier2') or {}
    for issue in t2.get('defects', []):
        p = 0 if issue['id'] in BLOCKING else 1
        out.append((p, issue['id'], tuple(sorted({g.owner[i] for i in issue.get('ids', []) if i in g.owner}))))
    for k, ids in sorted(((r.get('quality') or {}).get('defects') or {}).items()):
        if k == 'collisions':
            ids = [i for c in ids for i in c.get('ids', [])]
        owners = tuple(sorted({g.owner[i] for i in ids if isinstance(i, str) and i in g.owner}))
        if owners:
            out.append((1, 'quality.' + k, owners))
    for c in (r.get('diagnostics') or {}).get('collisions', []):
        out.append((1, 'collision', tuple(x for x in (c.get('a'), c.get('b')) if x in g.blocks)))
    for issue in t2.get('warnings', []):
        owners = tuple(sorted({g.owner[i] for i in issue.get('ids', []) if i in g.owner}))
        out.append((2, issue['id'], owners))
    seen, uniq = set(), []
    for p, code, owners in sorted(out, key=lambda t: (t[0], t[1], t[2])):
        k = (code, owners)
        if k not in seen:
            seen.add(k); uniq.append((p, code, owners))
    return uniq


# --------------------------------------------------------------- helpers
def _units(ctx, design):
    from .quality import units
    return units(ctx.analysis, design)


def _movable(ctx, design, ids: Iterable[str]):
    want = set(ids)
    return [(b, box) for b, box in _units(ctx, design)
            if b.id in want and b.role not in ('background', 'decoration') and not ctx.locked.intersection(b.ids)]


def _settle(ctx, design, frozen: Iterable[str] = ()):
    from .repair import contain, repair
    locked = set(ctx.locked) | set(frozen)
    out, _ = repair(ctx.source, contain(design, ctx.analysis, set(ctx.locked) | set(frozen)), ctx.analysis, locked)
    return _restore_locks(ctx, out)


def _restore_locks(ctx, design):
    by = {e['id']: e for e in design['elements']}
    for eid, rect in ctx.locks.items():
        if eid in by:
            by[eid].update({k: v for k, v in rect.items() if k in ('x', 'y', 'width', 'height')})
    return design


def _move(design, block, dx, dy):
    ids = set(block.ids)
    for e in design['elements']:
        if e['id'] in ids:
            e['x'] += dx; e['y'] += dy


def _scale(design, block, k, anchor):
    by = {e['id']: e for e in design['elements']}; ax, ay = anchor
    for i in block.ids:
        e = by[i]
        e.update(x=ax + (e['x'] - ax) * k, y=ay + (e['y'] - ay) * k, width=e['width'] * k, height=e['height'] * k)
        if e['type'] == 'text':
            e['font_size'] = max(1, round(e['font_size'] * k))


def _refit_block(ctx, design, block, fs_cap: float | None = None, grow: bool = True):
    """Largest font (<= source size) fitting each text's box; grows height when needed."""
    from .typography import fit_text, measure
    src = {e['id']: e for e in ctx.source['elements']}
    by = {e['id']: e for e in design['elements']}
    ch = float(design['canvas']['height'])
    for i in block.ids:
        e = by[i]
        if e['type'] != 'text' or i in ctx.locked:
            continue
        cap = int(min(src[i]['font_size'], fs_cap or src[i]['font_size']))
        fit = fit_text(e, e['width'], max(e['height'], 1), max(int(e['font_size']), 1))
        e['font_size'] = fit.font_size
        grown = measure(e, cap, e['width'])
        if grow and cap > e['font_size'] and not grown.overflow and grown.height <= min(ch, e['height'] * 1.6):
            e['font_size'] = cap; e['height'] = max(e['height'], grown.height)


def _push_apart(design, mover_box: Box, fixed: Box, block) -> tuple[float, float]:
    options = [(fixed.x1 - mover_box.x + 2, 0), (fixed.x - mover_box.x1 - 2, 0),
               (0, fixed.y1 - mover_box.y + 2), (0, fixed.y - mover_box.y1 - 2)]
    dx, dy = min(options, key=lambda v: abs(v[0]) + abs(v[1]))
    _move(design, block, dx, dy)
    return dx, dy


def _rows(blocks):
    rows = []
    for b, box in sorted(blocks, key=lambda t: (t[1].x, t[0].id)):
        for row in rows:
            if any(min(x.y1, box.y1) - max(x.y, box.y) > .5 * min(x.h, box.h) for _, x in row):
                row.append((b, box)); break
        else:
            rows.append([(b, box)])
    return [r for r in rows if len(r) >= 2]


def _columns(blocks):
    cols = []
    for b, box in sorted(blocks, key=lambda t: (t[1].y, t[0].id)):
        for col in cols:
            if any(min(x.x1, box.x1) - max(x.x, box.x) > .5 * min(x.w, box.w) for _, x in col):
                col.append((b, box)); break
        else:
            cols.append([(b, box)])
    return [c for c in cols if len(c) >= 2]


# ------------------------------------------------------ compound operators
# Each operator: op(ctx, r, nb, g) -> Iterator[(operation, design, reason)]

def op_solve_neighborhood(ctx, r, nb, g):
    """CP-SAT repair of the neighbourhood while every other group stays frozen."""
    from .constraints import solve
    design = r['design']
    frozen = {i for b, _ in _units(ctx, design) if b.id not in nb for i in b.ids}
    try:
        if _stop(ctx.budget):
            return
        remaining = ctx.budget.remaining() if ctx.budget is not None else .4
        out, event = solve(ctx.source, design, ctx.analysis, set(ctx.locked) | frozen,
                           budget_seconds=min(.4, remaining), cancelled=lambda: bool(_stop(ctx.budget)))
    except Exception as exc:  # optional solver: typed skip
        return
    if out:
        for e in out['elements']:
            if e['id'] in frozen:
                original = next(x for x in design['elements'] if x['id'] == e['id'])
                e.update({k: original[k] for k in MUTABLE_FIELDS if k in original})
        yield 'solve neighbourhood', _restore_locks(ctx, out), f"CP-SAT {event.get('status')} with {len(frozen)} frozen elements"


def op_widen_and_shift(ctx, r, nb, g):
    """Widen a cramped text block and shift its neighbours out of the way."""
    design = r['design']; cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    margin = max(8.0, .03 * min(cw, ch))
    blocks = _movable(ctx, design, nb)
    texts = [(b, box) for b, box in blocks if any(m['type'] == 'text' for m in b.members)]
    for b, box in texts[:2]:
        for f in (1.25, 1.5):
            trial = copy.deepcopy(design); by = {e['id']: e for e in trial['elements']}
            room = cw - margin - box.x
            if room <= box.w + 1:
                continue
            k = min(f, room / max(1, box.w))
            for i in b.ids:
                e = by[i]
                if e['type'] == 'text':
                    e['width'] = e['width'] * k
            _refit_block(ctx, trial, b)
            new = union_all(rotated_extent(by[i]) for i in b.ids)
            for o, obox in blocks:
                if o.id != b.id and obox.intersection_area(new) > 0:
                    _push_apart(trial, obox, new, o)
            yield f'widen {b.label.lower()} x{k:.2f} + shift neighbours', _settle(ctx, trial, ()), 'coordinated widen/shift'


def op_redistribute_row(ctx, r, nb, g):
    """Even out a row of blocks inside its free span and refit their labels.

    Locked (or otherwise fixed) row members bound the span; when the movable
    members do not fit, they shrink together and their text is refitted.
    """
    design = r['design']; cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    margin = max(8.0, .03 * min(cw, ch)); gap = max(4.0, .015 * cw)
    all_blocks = [(b, box) for b, box in _units(ctx, design) if b.role not in ('background', 'decoration')]
    for row in _rows(all_blocks):
        if not {b.id for b, _ in row}.intersection(nb):
            continue
        movable = sorted([(b, box) for b, box in row if not ctx.locked.intersection(b.ids)], key=lambda t: (t[1].x, t[0].id))
        fixed = [box for b, box in row if ctx.locked.intersection(b.ids)]
        if len(movable) < 2:
            continue
        first, last = movable[0][1], movable[-1][1]
        left = [f.x1 for f in fixed if f.cx < first.cx]; right = [f.x for f in fixed if f.cx > last.cx]
        lo = max(left) + gap if left else min(margin, first.x)
        hi = min(right) - gap if right else max(cw - margin, last.x1)
        widths = sum(box.w for _, box in movable); n = len(movable)
        room = hi - lo - gap * (n - 1)
        if room <= 0:
            continue
        k = min(1.0, room / max(1e-6, widths))
        trial = copy.deepcopy(design)
        placed = []
        for b, box in movable:
            if k < .999:
                _scale(trial, b, k, (box.x, box.y))
                _refit_block(ctx, trial, b, grow=False)
            placed.append((b, union_all(rotated_extent(e) for e in trial['elements'] if e['id'] in b.ids)))
        used = sum(box.w for _, box in placed)
        start, stop = (lo, hi) if k < .999 else (max(lo, first.x), min(hi, last.x1))
        step = (stop - start - used) / (n - 1)
        if step < 0:
            start, step = lo, gap
        x = start
        for b, box in placed:
            _move(trial, b, x - box.x, 0); x += box.w + step
        yield f'redistribute row of {n} (x{k:.2f}) + refit labels', _settle(ctx, trial), 'row redistribution'


def op_rebalance_column(ctx, r, nb, g):
    """Even gaps in a stacked column, recentre it and let panels restore padding."""
    design = r['design']; cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    blocks = [(b, box) for b, box in _units(ctx, design) if b.role not in ('background', 'decoration')]
    for col in _columns(blocks):
        if not set(b.id for b, _ in col).intersection(nb) or any(ctx.locked.intersection(b.ids) for b, _ in col):
            continue
        col = sorted(col, key=lambda t: (t[1].y, t[0].id))
        total = sum(box.h for _, box in col)
        top, bottom = col[0][1].y, max(box.y1 for _, box in col)
        gaps = len(col) - 1
        for mode in ('even', 'centre'):
            trial = copy.deepcopy(design)
            if mode == 'even':
                gap = max(2.0, (bottom - top - total) / gaps)
                y0 = top
            else:
                gap = max(2.0, min((bottom - top - total) / gaps, .04 * ch))
                y0 = (ch - (total + gap * gaps)) / 2
            y = y0
            for b, box in col:
                _move(trial, b, 0, y - box.y); y += box.h + gap
            yield f'rebalance column ({mode}) + restore padding', _settle(ctx, trial), 'column rebalance'


def op_scale_neighborhood(ctx, r, nb, g):
    design = r['design']
    blocks = _movable(ctx, design, nb)
    if not blocks:
        return
    u = union_all(box for _, box in blocks)
    for k in (.88, 1.12):
        trial = copy.deepcopy(design)
        for b, _ in blocks:
            _scale(trial, b, k, (u.cx, u.cy))
        yield f'scale neighbourhood x{k}', _settle(ctx, trial), 'joint neighbourhood scale'


def op_split_pair(ctx, r, nb, g):
    """Move both blocks of a colliding pair apart by half the overlap each."""
    design = r['design']
    blocks = {b.id: (b, box) for b, box in _movable(ctx, design, nb)}
    for c in (r.get('diagnostics') or {}).get('collisions', []):
        if c.get('a') not in blocks or c.get('b') not in blocks:
            continue
        (a, x), (b, y) = blocks[c['a']], blocks[c['b']]
        ox = min(x.x1, y.x1) - max(x.x, y.x); oy = min(x.y1, y.y1) - max(x.y, y.y)
        if ox <= 0 or oy <= 0:
            continue
        trial = copy.deepcopy(design)
        if ox < oy:
            s = 1 if x.cx <= y.cx else -1
            _move(trial, a, -s * (ox / 2 + 1), 0); _move(trial, b, s * (ox / 2 + 1), 0)
        else:
            s = 1 if x.cy <= y.cy else -1
            _move(trial, a, 0, -s * (oy / 2 + 1)); _move(trial, b, 0, s * (oy / 2 + 1))
        yield 'split colliding pair', _settle(ctx, trial), 'both blocks move half the overlap'


COMPOUND = (op_solve_neighborhood, op_widen_and_shift, op_redistribute_row, op_rebalance_column,
            op_split_pair, op_scale_neighborhood)


# -------------------------------------------------------- photo operators
def background_photo(ctx, design):
    """(element, source element) for the canvas background that may use cover."""
    from .scene import background_cover_allowed
    src = {e['id']: e for e in ctx.source['elements']}
    by = {e['id']: e for e in design['elements']}
    for b in getattr(ctx.analysis, 'background', []):
        if b['type'] == 'image' and b['id'] in by and background_cover_allowed(ctx.source, src[b['id']], {**by[b['id']], 'fit': 'cover'}):
            return by[b['id']], src[b['id']]
    return None, None


def op_cover_background(ctx, r, nb=None, g=None):
    """Fill the canvas with the background via the schema's centred cover crop.

    Variant 'anchored' keeps every element over the photo at the same point
    of the photograph (asset coordinates mapped through old and new fits).
    """
    from .photo_analysis import _asset_path, fit_mapping
    design = r['design']; cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    bg, _ = background_photo(ctx, design)
    if bg is None or bg['id'] in ctx.locked:
        return
    if bg.get('fit') == 'cover' and abs(bg['x']) < .5 and abs(bg['y']) < .5 and abs(bg['width'] - cw) < .5 and abs(bg['height'] - ch) < .5:
        return
    trial = copy.deepcopy(design)
    tb = next(e for e in trial['elements'] if e['id'] == bg['id'])
    tb.update(x=0.0, y=0.0, width=cw, height=ch, fit='cover')
    yield 'cover canvas with background', _settle(ctx, trial), 'background fills canvas (centred cover crop)'
    path = _asset_path(bg.get('asset_id'))
    from .structure import flat_asset
    if path is None or flat_asset(bg.get('asset_id')):
        return
    from PIL import Image
    with Image.open(path) as im:
        size = im.size
    sx0, sy0, ox0, oy0, clip0 = fit_mapping(bg, size)
    sx1, sy1, ox1, oy1, _ = fit_mapping(tb, size)
    trial2 = copy.deepcopy(trial)
    by = {e['id']: e for e in trial2['elements']}
    for b, box in _units(ctx, design):
        if ctx.locked.intersection(b.ids) or clip0.intersection_area(box) < .5 * max(1, box.area):
            continue
        u, v = (box.cx - ox0) / sx0, (box.cy - oy0) / sy0
        nx, ny = u * sx1 + ox1, v * sy1 + oy1
        for i in b.ids:
            by[i]['x'] += nx - box.cx; by[i]['y'] += ny - box.cy
    yield 'cover background + keep copy anchored to the photo', _settle(ctx, trial2), 'photo-anchored remap'


def op_quiet_text(ctx, r, nb=None, g=None):
    """Move text blocks off important photo subjects into quiet, legible regions."""
    from .photo_analysis import color_luminance, design_field, quiet_regions
    if ctx.photo_mode != 'subject_aware':
        return
    design = r['design']
    hosts = _source_hosts(ctx.source, ctx.analysis)
    bg, _ = background_photo(ctx, design)
    from .photo_analysis import photo_hosts
    photos = {e['id'] for e in photo_hosts(design, ctx.analysis)}
    if not photos:
        return
    fields = {}
    blocks = _units(ctx, design)
    others = {b.id: box for b, box in blocks}
    from .quality import ink
    by = {e['id']: e for e in design['elements']}
    moved = 0
    for b, box in blocks:
        if ctx.locked.intersection(b.ids) or not any(m['type'] == 'text' for m in b.members):
            continue
        host = next((hosts[i] for i in b.ids if i in hosts and hosts[i] in photos), None)
        if host is None and bg is not None and bg['id'] in photos:
            host = bg['id']
        if host is None:
            continue
        if host not in fields:
            fields[host] = design_field(ctx.source, design, host, mode=ctx.photo_mode)
        f = fields[host]
        if f is None:
            continue
        overlap = max(f.subject_overlap(ink(by[i])) for i in b.ids if by[i]['type'] == 'text')
        if overlap < .08:
            continue
        hb = element_box(by[host]).clip(f.cw, f.ch)
        text = next(m for m in b.members if m['type'] == 'text')
        avoid = [o for k, o in others.items() if k != b.id]
        regions = quiet_regions(f, box.w, box.h, text_lum=color_luminance(text.get('color')), within=hb,
                                avoid=avoid, prefer=(box.cx, box.cy), max_results=2)
        for q in regions:
            trial = copy.deepcopy(design)
            _move(trial, b, q.box.x - box.x, q.box.y - box.y)
            moved += 1
            yield f'move {b.label.lower()} to quiet photo region', _settle(ctx, trial), f'subject overlap {overlap:.2f} -> {q.subject_overlap:.2f}'
        if moved >= 4:
            return


PHOTO = (op_cover_background, op_quiet_text)


# --------------------------------------------------------- operator stats
@dataclass
class OperatorStats:
    """Improvement per unit cost per operator family, with UCB exploration."""
    version: str = EDIT_VERSION
    tries: dict[str, int] = field(default_factory=dict)
    gain: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)

    def record(self, family: str, gain: float, cost: float) -> None:
        self.tries[family] = self.tries.get(family, 0) + 1
        self.gain[family] = self.gain.get(family, 0.0) + max(0.0, gain)
        self.cost[family] = self.cost.get(family, 0.0) + max(1e-3, cost)

    def duplicate(self, family: str) -> None:
        self.duplicates[family] = self.duplicates.get(family, 0) + 1

    def score(self, family: str) -> float:
        n = self.tries.get(family, 0); total = sum(self.tries.values())
        prior = PRIORS.get(family.split(':')[0], 1.0)
        mean = (self.gain.get(family, 0.0) + prior) / (self.cost.get(family, 0.0) + 1.0)
        return mean + .7 * math.sqrt(math.log(total + 2) / (n + 1))

    def to_dict(self):
        return {'version': self.version, 'tries': dict(sorted(self.tries.items())),
                'gain': {k: round(v, 3) for k, v in sorted(self.gain.items())},
                'duplicates': dict(sorted(self.duplicates.items()))}


# Prior improvement per family before any evidence: the photo pass first (it
# changes the whole scene), then the proven legacy operators, then compounds.
PRIORS = {'photo': 3.0, 'legacy': 2.0, 'compound': 1.0}
_STATS: dict[str, OperatorStats] = {}


def operator_stats(version: str = EDIT_VERSION, reset: bool = False) -> OperatorStats:
    if reset or version not in _STATS:
        _STATS[version] = OperatorStats(version)
    return _STATS[version]


# ------------------------------------------------------ edit generation
def _legacy_operators():
    from .refine import OPERATORS
    return OPERATORS


def iter_edits(ctx: EditContext, r: dict[str, Any], *, stats: OperatorStats | None = None,
               include_legacy: bool = True, include_compound: bool = True, include_photo: bool | None = None,
               max_neighborhoods: int = 4) -> Iterator[tuple[GeometryEdit, dict[str, Any]]]:
    """Yield (typed edit, materialized design) for a candidate, most useful first."""
    design = r['design']
    stats = stats or OperatorStats()
    include_photo = ctx.photo_mode == 'subject_aware' if include_photo is None else include_photo
    groups: list[tuple[str, Callable[[], Iterable]]] = []
    if include_photo:
        for op in PHOTO:
            groups.append(('photo:' + op.__name__[3:], lambda op=op: op(ctx, r)))
    if include_compound:
        g = build_graph(ctx, r)
        hoods, seen = [], set()
        for _, code, owners in defect_seeds(r, g):
            if not owners:
                continue
            nb = g.neighborhood(owners)
            if frozenset(nb) in seen:
                continue
            seen.add(frozenset(nb)); hoods.append((code, nb))
            if len(hoods) >= max_neighborhoods:
                break
        for code, nb in hoods:
            for op in COMPOUND:
                if op is op_solve_neighborhood and not ctx.allow_solver:
                    continue
                groups.append((f'compound:{op.__name__[3:]}', lambda op=op, nb=nb, code=code: ((n, d, f'{why}; defect {code}', nb) for n, d, why in op(ctx, r, set(nb), g))))
    if include_legacy:
        locked = set(ctx.locked)
        for op in _legacy_operators():
            groups.append(('legacy:' + op.__name__[3:], lambda op=op: ((n, d, 'legacy operator') for n, d in op(ctx.source, r, ctx.analysis, locked))))
    # Adaptive scheduling: families with higher improvement-per-cost go first,
    # with UCB exploration so an unlucky family is never starved. Stable order.
    order = sorted(range(len(groups)), key=lambda n: (-round(stats.score(groups[n][0]), 6), n))
    for n in order:
        family, make = groups[n]
        if _stop(ctx.budget):
            return
        last = time.monotonic()
        try:
            for item in make():
                now = time.monotonic(); cost, last = now - last, now
                if len(item) == 4:
                    name, out, why, nb = item
                else:
                    (name, out, why), nb = item, ()
                try:
                    edit = diff_edit(design, out, name, family=family, reason=why, cost=cost, neighborhood=nb)
                except ValueError:
                    continue
                if edit is not None:
                    yield edit, apply(design, edit)
                if _stop(ctx.budget):
                    return
                last = time.monotonic()
        except (ValueError, KeyError, ZeroDivisionError, IndexError, OSError):
            continue


def materializable(source, design_before, design_after, locks, operation='edit') -> bool:
    """True when Codex's materializer accepts the edit (geometry allowlist, locks, validity)."""
    from .contracts import DesignRevision
    from .materialize import apply_edit, proposal_from_design
    try:
        rev = DesignRevision.create(design_before)
        apply_edit(source, rev, proposal_from_design(rev, design_after, operation), locks)
        return True
    except (ValueError, KeyError, TypeError):
        return False


def expand(candidate: dict[str, Any], budget: Any = None, *, source: dict[str, Any], analysis: Any,
           locks: Any = None, photo_mode: str | None = None, max_edits: int | None = None,
           allow_solver: bool = True) -> Iterator[tuple[str, dict[str, Any]]]:
    """Controller expansion: ``(operation_name, design)`` pairs for ``run_controller(expand=...)``.

    Every yielded design differs from ``candidate['design']`` only in
    allowlisted geometry and is pre-checked with ``materialize.apply_edit``
    against the same locks, so the controller never receives an edit it must
    reject for structural reasons. Duplicate geometry is not yielded twice.
    """
    if max_edits is not None and max_edits <= 0:
        return
    ctx = EditContext.create(source, analysis, locks, budget, candidate['design'], photo_mode)
    ctx.allow_solver = allow_solver
    seen = {design_hash(candidate['design'])}
    count = 0
    for edit, design in iter_edits(ctx, candidate):
        if _stop(budget):
            return
        h = design_hash(design)
        if h in seen:
            continue
        seen.add(h)
        if not materializable(source, candidate['design'], design, ctx.locks, edit.operation):
            continue
        yield edit.operation, design
        count += 1
        if max_edits is not None and count >= max_edits:
            return
