"""Evaluate-and-fix loop: improve the best first-pass candidates until the rubric stops improving.

Each round reads the candidate's rubric issues (engine/rubric.py), applies the
repair operators that target them, rescores every trial with the same
pipeline scoring, and keeps the single best improvement. Acceptance is
strictly monotonic on (gate, fewer hard defects, higher overall, fewer
warnings), so a refined candidate can never rank below its seed. Operators
only change x, y, width, height and font_size; validation still runs on
every trial.
"""
from __future__ import annotations

import copy, math, time
from .blocks import FLOORS
from .canonical import Box, element_box, union_all
from .quality import units
from .repair import contain, move, repair
from .typography import fit_text, measure


# Defects the release gate blocks on: never traded for fewer of the others.
BLOCKING = ('overlap.text', 'grouping.rigid', 'legibility.overflow')


def blocking(r):
    return sum(x['id'] in BLOCKING for x in (r['rubric']['tier2'] or {}).get('defects', []))


def key(r):
    rub = r['rubric']; t2 = rub['tier2'] or {}
    hard = r['quality']['defect_count'] + len(t2.get('defects', []))
    return (bool(r['valid'] and rub['gate']), -blocking(r), -hard, round(t2.get('overall', 0), 1), -len(t2.get('warnings', [])))


def _issues(r, prefix):
    t2 = r['rubric']['tier2'] or {}
    return [x for x in t2.get('defects', []) + t2.get('warnings', []) if x['id'].startswith(prefix)]


def _ids(r, prefix):
    return {i for x in _issues(r, prefix) for i in x['ids']}


def _blocks(A, design, locked):
    return [(b, box) for b, box in units(A, design) if b.role not in ('background', 'decoration') and not locked.intersection(b.ids)]


def _scale_block(design, block, k, anchor):
    by = {e['id']: e for e in design['elements']}; ax, ay = anchor
    for i in block.ids:
        e = by[i]
        e.update(x=ax + (e['x'] - ax) * k, y=ay + (e['y'] - ay) * k, width=e['width'] * k, height=e['height'] * k)
        if e['type'] == 'text':
            e['font_size'] = max(1, round(e['font_size'] * k))


def _settle(source, design, A, locked):
    out, _ = repair(source, contain(design, A, locked), A, locked)
    return out


def op_collisions(source, r, A, locked):
    if _issues(r, 'overlap') or r['diagnostics']['collisions']:
        yield 'resolve collisions', _settle(source, copy.deepcopy(r['design']), A, locked)


def op_legibility(source, r, A, locked):
    """Grow too-small text: in place (wider, taller box) or by scaling its whole block."""
    ids = (_ids(r, 'legibility') | set(r['quality']['defects']['unreadable'])) - locked
    if not ids:
        return
    src = {e['id']: e for e in source['elements']}; cw, ch = r['design']['canvas']['width'], r['design']['canvas']['height']
    owner = {i: b for b in A.all_blocks() for i in b.ids}
    margin = max(8.0, .03 * min(cw, ch))
    trial = copy.deepcopy(r['design']); by = {e['id']: e for e in trial['elements']}; changed = False
    for i in sorted(ids)[:4]:
        e = by.get(i)
        if not e or e['type'] != 'text':
            continue
        want = min(src[i]['font_size'], max(FLOORS.get(A.roles.get(i, 'body'), 9), e['font_size'] + 1))
        width = min(cw - margin - e['x'], max(e['width'], e['width'] * 1.4))
        fit = measure(e, int(want), max(1, width))
        e.update(width=max(e['width'], width), height=max(e['height'], fit.height), font_size=int(want)); changed = True
    if changed:
        yield 'enlarge small text', _settle(source, trial, A, locked)
    for b in {owner[i].id: owner[i] for i in ids if i in owner}.values():
        box = next((x for u, x in units(A, r['design']) if u.id == b.id), None)
        if not box:
            continue
        for k in (1.15, 1.35):
            trial = copy.deepcopy(r['design']); _scale_block(trial, b, k, (box.x, box.y))
            yield f'scale {b.label.lower()} x{k}', _settle(source, trial, A, locked)


def op_hierarchy(source, r, A, locked):
    ids = _ids(r, 'hierarchy.inversion') - locked
    if not ids:
        return
    src = {e['id']: e for e in source['elements']}; trial = copy.deepcopy(r['design']); by = {e['id']: e for e in trial['elements']}
    texts = sorted((i for i in ids if by.get(i, {}).get('type') == 'text'), key=lambda i: -src[i]['font_size'])
    for n, hi in enumerate(texts):
        for lo in texts[n + 1:]:
            if src[hi]['font_size'] > 1.15 * src[lo]['font_size'] and by[lo]['font_size'] >= by[hi]['font_size']:
                by[lo]['font_size'] = max(1, by[hi]['font_size'] - 1)
    yield 'restore type size order', _settle(source, trial, A, locked)


def op_align(source, r, A, locked):
    """Snap near-miss edges (1 px to 1% apart) onto the larger block's line."""
    design = r['design']; cw, ch = design['canvas']['width'], design['canvas']['height']
    if not _issues(r, 'alignment'):
        return
    trial = copy.deepcopy(design); moved = set()
    blocks = sorted(_blocks(A, trial, locked), key=lambda t: -t[1].area)
    for n, (b, box) in enumerate(blocks):
        if b.id in moved:
            continue
        for anchor, abox in blocks[:n]:
            dx = next((t - s for s, t in ((box.x, abox.x), (box.cx, abox.cx), (box.x1, abox.x1)) if 1 < abs(t - s) <= .01 * cw), 0)
            dy = next((t - s for s, t in ((box.y, abox.y), (box.cy, abox.cy), (box.y1, abox.y1)) if 1 < abs(t - s) <= .01 * ch), 0)
            if dx or dy:
                move(trial, b, dx, dy); moved.add(b.id); break
    if moved:
        yield 'snap near-miss alignments', contain(trial, A)


def op_space(source, r, A, locked):
    design = r['design']; cw, ch = design['canvas']['width'], design['canvas']['height']
    margin = max(8.0, .03 * min(cw, ch)); blocks = _blocks(A, design, locked)
    if not blocks:
        return
    tight = _ids(r, 'space.margin')
    if tight:
        trial = copy.deepcopy(design)
        for b, box in blocks:
            if b.id in tight:
                move(trial, b, max(0, margin - box.x) - max(0, box.x1 - (cw - margin)), max(0, margin - box.y) - max(0, box.y1 - (ch - margin)))
        yield 'pull content inside margins', _settle(source, trial, A, locked)
    content = union_all(box for _, box in blocks)
    stats = (r['rubric']['tier2'] or {}).get('stats', {})
    if stats.get('balance', 1) < .9 or _issues(r, 'space'):
        trial = copy.deepcopy(design)
        for b, _ in blocks:
            move(trial, b, cw / 2 - content.cx, ch / 2 - content.cy)
        yield 'centre content', contain(trial, A)
    if _issues(r, 'space.empty') or _issues(r, 'space.dead_zone') or _issues(r, 'legibility'):
        room = min((cw - 2 * margin) / max(1, content.w), (ch - 2 * margin) / max(1, content.h))
        for k in sorted({min(room, 1.1), min(room, 1.25)}):
            if k <= 1.01:
                continue
            trial = copy.deepcopy(design)
            for b, _ in blocks:
                _scale_block(trial, b, k, (content.cx, content.cy))
            for b, _ in blocks:
                move(trial, b, cw / 2 - content.cx, ch / 2 - content.cy)
            yield f'grow content x{k:.2f}', _settle(source, trial, A, locked)
    if _issues(r, 'space.uneven_gaps'):
        trial = copy.deepcopy(design)
        col = sorted(_blocks(A, trial, locked), key=lambda t: t[1].y)
        stack = [col[0]] if col else []
        for b, box in col[1:]:
            last = stack[-1][1]
            if min(last.x1, box.x1) - max(last.x, box.x) > .5 * min(last.w, box.w) and box.y >= last.y1:
                stack.append((b, box))
        if len(stack) >= 3:
            gap = (stack[-1][1].y - stack[0][1].y1 - sum(box.h for _, box in stack[1:-1])) / (len(stack) - 1)
            y = stack[0][1].y1
            for b, box in stack[1:-1]:
                y += gap; move(trial, b, 0, y - box.y); y += box.h
            yield 'even out vertical gaps', contain(trial, A)


def op_groups(source, r, A, locked):
    broken = _ids(r, 'grouping.rigid') - locked
    if not broken:
        return
    src = {e['id']: e for e in source['elements']}; trial = copy.deepcopy(r['design']); by = {e['id']: e for e in trial['elements']}
    for b in A.all_blocks():
        if not broken.intersection(b.ids):
            continue
        s0 = union_all(Box(src[i]['x'], src[i]['y'], src[i]['width'], src[i]['height']) for i in b.ids)
        s1 = union_all(Box(by[i]['x'], by[i]['y'], by[i]['width'], by[i]['height']) for i in b.ids)
        k = min(s1.w / max(1, s0.w), s1.h / max(1, s0.h))
        for i in b.ids:
            e, o = src[i], by[i]
            o.update(x=s1.x + (e['x'] - s0.x) * k, y=s1.y + (e['y'] - s0.y) * k, width=e['width'] * k, height=e['height'] * k)
            if o['type'] == 'text':
                o['font_size'] = fit_text(o, o['width'], o['height'], max(1, round(e['font_size'] * k))).font_size
    yield 'restore group layout', _settle(source, trial, A, locked)


def op_imagery(source, r, A, locked):
    ids = _ids(r, 'imagery.presence') - locked
    for b, box in _blocks(A, r['design'], locked):
        if ids.intersection(b.ids):
            trial = copy.deepcopy(r['design']); _scale_block(trial, b, 1.25, (box.x, box.y))
            yield f'enlarge {b.label.lower()}', _settle(source, trial, A, locked)


def _columns(blocks):
    """Groups of blocks stacked in a shared column (horizontal overlap > half the narrower)."""
    cols = []
    for b, box in sorted(blocks, key=lambda t: t[1].y):
        for col in cols:
            if any(min(x.x1, box.x1) - max(x.x, box.x) > .5 * min(x.w, box.w) for _, x in col):
                col.append((b, box)); break
        else:
            cols.append([(b, box)])
    return [c for c in cols if len(c) >= 2]


def op_axis(source, r, A, locked):
    """Give stacked blocks one shared left edge or centre line (alignment, NN/g order)."""
    score = ((r['rubric']['tier2'] or {}).get('scores') or {}).get('alignment', 100)
    if score >= 95:
        return
    for mode in ('centre', 'left'):
        trial = copy.deepcopy(r['design']); changed = False
        for col in _columns(_blocks(A, trial, locked)):
            ref = max(col, key=lambda t: t[1].w)[1]
            for b, box in col:
                dx = (ref.cx - box.cx) if mode == 'centre' else (ref.x - box.x)
                if abs(dx) > .5:
                    move(trial, b, dx, 0); changed = True
        if changed:
            yield f'align columns ({mode})', _settle(source, trial, A, locked)


def _rows(blocks):
    rows = []
    for b, box in sorted(blocks, key=lambda t: t[1].x):
        for row in rows:
            if any(min(x.y1, box.y1) - max(x.y, box.y) > .5 * min(x.h, box.h) for _, x in row):
                row.append((b, box)); break
        else:
            rows.append([(b, box)])
    return [r for r in rows if len(r) >= 2]


def op_rows(source, r, A, locked):
    """Side-by-side blocks share a top edge or a middle line."""
    if ((r['rubric']['tier2'] or {}).get('scores') or {}).get('alignment', 100) >= 95:
        return
    for mode in ('middle', 'top'):
        trial = copy.deepcopy(r['design']); changed = False
        for row in _rows(_blocks(A, trial, locked)):
            ref = max(row, key=lambda t: t[1].h)[1]
            for b, box in row:
                dy = (ref.cy - box.cy) if mode == 'middle' else (ref.y - box.y)
                if abs(dy) > .5:
                    move(trial, b, 0, dy); changed = True
        if changed:
            yield f'align rows ({mode})', _settle(source, trial, A, locked)


def op_comfort(source, r, A, locked):
    """Readable is not comfortable: grow the text blocks furthest below their source size."""
    if ((r['rubric']['tier2'] or {}).get('scores') or {}).get('legibility', 100) >= 90:
        return
    src = {e['id']: e for e in source['elements']}
    texty = [(b, box) for b, box in _blocks(A, r['design'], locked) if any(m['type'] == 'text' for m in b.members)]
    by = {e['id']: e for e in r['design']['elements']}
    def shrink(t):
        b = t[0]; return min(by[i]['font_size'] / max(1, src[i]['font_size']) for i in b.ids if by[i]['type'] == 'text')
    texty.sort(key=shrink)
    for k in (1.12, 1.25):
        trial = copy.deepcopy(r['design'])
        for b, box in texty[:3]:
            _scale_block(trial, b, k, (box.x, box.y))
        yield f'grow text blocks x{k}', _settle(source, trial, A, locked)


def op_safe_zone(source, r, A, locked):
    from .rubric import safe_zones
    ids = _ids(r, 'space.safe_zone')
    if not ids:
        return
    design = r['design']; cw, ch = design['canvas']['width'], design['canvas']['height']; zones = safe_zones(int(cw), int(ch))
    trial = copy.deepcopy(design)
    for b, box in _blocks(A, trial, locked):
        if b.id not in ids:
            continue
        for z in zones:
            if z.intersection_area(box) <= 0:
                continue
            # Smallest push that clears this zone along one axis.
            options = [(z.x1 - box.x, 0), (z.x - box.x1, 0), (0, z.y1 - box.y), (0, z.y - box.y1)]
            dx, dy = min(options, key=lambda v: abs(v[0]) + abs(v[1]))
            move(trial, b, dx, dy); box = Box(box.x + dx, box.y + dy, box.w, box.h)
    yield 'clear platform UI zones', _settle(source, trial, A, locked)


def op_measure(source, r, A, locked):
    ids = _ids(r, 'legibility.measure') - locked
    if not ids:
        return
    trial = copy.deepcopy(r['design']); by = {e['id']: e for e in trial['elements']}
    for i in ids:
        e = by.get(i)
        if not e or e['type'] != 'text':
            continue
        fit = measure(e, int(e['font_size']), max(1, e['width']))
        per_line = len(e['content']) / max(1, fit.lines)
        if per_line > 75:
            e['width'] *= 70 / per_line
            e['height'] = max(e['height'], measure(e, int(e['font_size']), e['width']).height)
    yield 'shorten long lines', _settle(source, trial, A, locked)


OPERATORS = (op_collisions, op_legibility, op_comfort, op_hierarchy, op_groups, op_align, op_axis, op_rows, op_space, op_safe_zone, op_measure, op_imagery)


REFINEMENT_MODES = ('hill_climb', 'beam_lns')


def refinement_mode(explicit=None):
    """`hill_climb` (legacy) unless explicitly or env-selected (CHAMELEON_REFINEMENT_MODE)."""
    import os
    value = explicit if explicit is not None else os.environ.get('CHAMELEON_REFINEMENT_MODE', 'hill_climb')
    return value if value in REFINEMENT_MODES else 'hill_climb'


def refine(seed, assess, source, A, locked=(), max_rounds=10, deadline=None, target=95.0, *, mode=None, budget=None,
           operators=None, photo_mode=None, allow_solver=True):
    """Improve one candidate. `assess(design)` returns a scored result or None (duplicate).

    Legacy adapter: positional signature unchanged. ``mode`` (or
    CHAMELEON_REFINEMENT_MODE) selects the strict hill climb (default) or the
    bounded beam / large-neighbourhood search of ``beam_refine``.
    """
    from .photo_analysis import photo_mode as _photo_mode
    if refinement_mode(mode) == 'beam_lns':
        return beam_refine(seed, assess, source, A, locked, max_depth=min(6, max(1, max_rounds)), deadline=deadline,
                           target=target, budget=budget, operators=operators, photo_mode=photo_mode, allow_solver=allow_solver)
    if _photo_mode(photo_mode) == 'subject_aware' and operators is None:
        from .edit_operators import EditContext, PHOTO
        ctx = EditContext.create(source, A, locked, budget, seed['design'], 'subject_aware')
        extra = tuple((lambda op: lambda s, r, a, l: ((n, d) for n, d, _ in op(ctx, r)))(op) for op in PHOTO)
        return _hill_climb(seed, assess, source, A, locked, max_rounds, deadline, target, OPERATORS + extra, budget)
    return _hill_climb(seed, assess, source, A, locked, max_rounds, deadline, target, operators or OPERATORS, budget)


def _hill_climb(seed, assess, source, A, locked, max_rounds, deadline, target, operators, budget=None):
    from .edit_operators import _stop
    locked = set(locked); current = seed; steps = []
    for round_ in range(max_rounds):
        t2 = current['rubric']['tier2'] or {}
        if key(current)[:3] == (True, 0, 0) and (not t2.get('warnings') or t2.get('overall', 0) >= target):
            break  # clean and at target: further polishing costs time for little gain
        best = None
        for op in operators:
            for name, design in op(source, current, A, locked):
                if _stop(budget) or (deadline and time.monotonic() > deadline):
                    return current, steps
                edits_before = getattr(budget, 'edits', None)
                trial = assess(design)
                if budget is not None and hasattr(budget, 'charge'):budget.charge()
                elif edits_before is not None and budget.edits == edits_before:budget.edits += 1
                if trial and key(trial) > key(best[0] if best else current):
                    best = (trial, name)
        if not best:
            break
        before, current = current, best[0]
        steps.append({'refine': best[1], 'round': round_ + 1, 'overall': [key(before)[3], key(current)[3]], 'hard_defects': [-key(before)[2], -key(current)[2]]})
        if deadline and time.monotonic() > deadline:
            break
    return current, steps


# ----------------------------------------------------------- beam / LNS
def utility(r):
    """Scalar summary of `key` for operator statistics only (never for acceptance)."""
    k = key(r)
    return 1000 * k[0] + 100 * k[1] + 20 * k[2] + k[3] + .5 * k[4]


def _visual_penalty(ctx, r):
    """Photo terms the rubric does not see: copy on a subject, unpainted canvas bands."""
    if ctx.photo_mode != 'subject_aware':
        return 0.0
    cache = ctx._photo.setdefault('visual', {})
    if id(r) in cache:
        return cache[id(r)][1]
    from .photo_analysis import fit_mapping, text_on_subject, _asset_path
    from .edit_operators import background_photo
    design = r['design']; cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    on = text_on_subject(ctx.source, design, mode=ctx.photo_mode)
    pen = 10.0 * sum(on.values()) / max(1, sum(e['type'] == 'text' for e in design['elements']))
    bg, _ = background_photo(ctx, design)
    if bg is not None and _asset_path(bg.get('asset_id')) is not None:
        from PIL import Image
        with Image.open(_asset_path(bg.get('asset_id'))) as im:
            clip = fit_mapping(bg, im.size)[4].clip(cw, ch)
        pen += 5.0 * (1 - clip.area / (cw * ch))
    cache[id(r)] = (r, round(pen, 4))  # keep r alive so id() stays unique
    return cache[id(r)][1]


def ambient_bites(ctx, r):
    """Texts newly running into a decoration/panel edge (the rubric treats panels as
    ambient). Sitting fully inside a panel, or a bite already in the source, is fine."""
    cache = ctx._photo.setdefault('bites', {})
    if id(r) in cache:
        return cache[id(r)][1]
    from .quality import ink
    from .rubric import _bites
    A = ctx.analysis; design = r['design']
    src = {e['id']: e for e in ctx.source['elements']}; out = {e['id']: e for e in design['elements']}
    cw, ch = float(design['canvas']['width']), float(design['canvas']['height'])
    panels = [e['id'] for e in A.decorations + A.background
              if element_box(out[e['id']]).clip(cw, ch).area < .85 * cw * ch]
    n = 0
    for t in out.values():
        if t['type'] != 'text' or not str(t.get('content', '')).strip():
            continue
        tb = ink(t)
        for pid in panels:
            if out[pid].get('z_index', 0) >= t.get('z_index', 0):
                continue
            if out[pid]['type'] == 'image':
                from .photo_analysis import painted_coverage
                if painted_coverage(out[pid], tb, design) <= .05:
                    continue
            pb = element_box(out[pid])
            inside = pb.intersection_area(tb) >= .95 * max(1e-6, tb.area)
            if _bites(tb, pb) and not inside and not _bites(ink(src[t['id']]), element_box(src[pid])):
                n += 1
    cache[id(r)] = (r, n)
    return n


def search_key(ctx, r):
    k = key(r)
    return (k[0], k[1], k[2] - ambient_bites(ctx, r)) + k[3:] + (-_visual_penalty(ctx, r),)


def _hard_valid(r):
    return bool(r and r['valid'] and r['rubric']['gate'])


def _cheap(ctx, design):
    """Structural and text checks without rendering or rubric: (invalid, blocking-ish count)."""
    from .invariants import passes, validate
    from .quality import collisions
    from .typography import measure
    if not passes(validate(ctx.source, design)):
        return (1, 0)
    for eid, rect in ctx.locks.items():
        e = next((x for x in design['elements'] if x['id'] == eid), None)
        if e is None or any(abs(e[k] - v) > 1e-6 for k, v in rect.items() if k in ('x', 'y', 'width', 'height')):
            return (1, 0)
    over = sum(1 for e in design['elements'] if e['type'] == 'text' and (lambda f: f.overflow or f.height > e['height'] + .5)(measure(e, int(e['font_size']), max(1, e['width']))))
    return (0, len(collisions(ctx.source, design, ctx.analysis)) + over)


def beam_refine(seed, assess, source, A, locked=(), *, max_depth=6, width=4, horizon=6, patience=2, deadline=None,
                target=95.0, budget=None, operators=None, photo_mode=None, stats=None, allow_solver=True):
    """Small beam of hard-valid states over typed (compound) edits.

    * every trial is deduplicated by materialized geometry (cycle detection);
    * only hard-valid assessed states enter the beam, so softer quality may
      dip temporarily while the best assessed incumbent is kept separately;
    * the incumbent - never the last explored state - is returned;
    * stops on target, exhausted neighbourhoods, budget, deadline, cancellation,
      ``patience`` depths without a better incumbent (stagnation, which leaves
      the shared time to later seeds as the hill climb does), more than
      ``horizon`` depths past the last improvement, or ``max_depth``.

    ``operators`` (tests/ablations) replaces the edit generator with callables
    ``op(ctx, candidate) -> iterable of (name, design)``.
    """
    from .edit_operators import EditContext, SearchBudget, design_hash, diff_edit, iter_edits, OperatorStats, materializable, _stop
    budget = budget or SearchBudget(deadline=deadline)
    ctx = EditContext.create(source, A, locked, budget, seed['design'], photo_mode)
    ctx.allow_solver = allow_solver
    stats = stats or OperatorStats()
    visited = {design_hash(seed['design'])}
    incumbent, beam = seed, [seed]
    parents = {id(seed): None}
    stop, depth, evaluated, last_gain = None, 0, 0, 0
    eval_cost = []

    def edits(state):
        if operators is None:
            yield from iter_edits(ctx, state, stats=stats)
            return
        for op in operators:
            for name, design in op(ctx, state):
                try:
                    e = diff_edit(state['design'], design, name, family=getattr(op, '__name__', 'custom'))
                except ValueError:
                    continue
                if e is not None:
                    yield e, design

    for depth in range(max_depth):
        t2 = incumbent['rubric']['tier2'] or {}
        if key(incumbent)[:3] == (True, 0, 0) and (not t2.get('warnings') or t2.get('overall', 0) >= target) and not _visual_penalty(ctx, incumbent):
            stop = 'target'; break
        pool = []
        before = incumbent
        for state in beam:
            stop = _stop(budget)
            if stop:
                break
            base = _cheap(ctx, state['design'])
            # Greedy while improving (one state, fully expanded like the hill
            # climb); a widened beam explores each alternative more shallowly.
            cap = 10 ** 6 if len(beam) == 1 else 8
            # Edits stream lazily; each is deduplicated and cheaply screened
            # (validity, locks, collisions, overflow) before it may pay for a
            # full assessment. Screened-out trials wait in reserve.
            reserve, spent = [], 0
            def run(edit, design):
                nonlocal incumbent, evaluated
                if _stop(budget) or not materializable(source, state['design'], design, ctx.locks, edit.operation):
                    return
                started = time.monotonic()
                edits_before = getattr(budget, 'edits', None)
                trial = assess(design)
                if hasattr(budget, 'charge'):
                    budget.charge()
                elif edits_before is not None and budget.edits == edits_before:
                    budget.edits += 1
                eval_cost.append(time.monotonic() - started); evaluated += 1
                if not _hard_valid(trial):
                    stats.record(edit.family, 0.0, edit.estimated_cost + eval_cost[-1])
                    return
                stats.record(edit.family, utility(trial) - utility(state), edit.estimated_cost + eval_cost[-1])
                parents[id(trial)] = (state, edit)
                pool.append(trial)
                if search_key(ctx, trial) > search_key(ctx, incumbent):
                    incumbent = trial
            for edit, design in edits(state):
                stop = _stop(budget)
                if stop or spent >= cap:
                    break
                h = design_hash(design)
                if h in visited:
                    stats.duplicate(edit.family); continue
                visited.add(h)
                c = _cheap(ctx, design)
                if c[0]:
                    continue
                if c[1] > base[1]:
                    reserve.append((c[1], len(reserve), edit, design)); continue
                run(edit, design); spent += 1
            for _, _, edit, design in sorted(reserve, key=lambda t: (t[0], t[1])):
                stop = stop or _stop(budget)
                if stop or spent >= cap:
                    break
                run(edit, design); spent += 1
            if stop:
                break
        if stop:
            break
        if not pool:
            stop = 'exhausted'; break
        if incumbent is not before:
            last_gain = depth + 1
            beam = [incumbent]
            continue
        if depth + 1 - last_gain >= min(patience, horizon):
            stop = 'stagnation'; break
        # No improvement: widen. Diverse selection of hard-valid states (soft
        # quality may dip), best first, at most two per operator family.
        pool.sort(key=lambda t: search_key(ctx, t), reverse=True)
        beam, families = [], {}
        for t in pool:
            fam = parents[id(t)][1].family
            if families.get(fam, 0) >= 2:
                continue
            families[fam] = families.get(fam, 0) + 1
            beam.append(t)
            if len(beam) >= width:
                break
    else:
        stop = 'max_depth'
    steps, node = [], incumbent
    chain = []
    while parents.get(id(node)):
        parent, edit = parents[id(node)]
        chain.append((parent, node, edit)); node = parent
    for n, (before, after, edit) in enumerate(reversed(chain)):
        steps.append({'refine': edit.operation, 'round': n + 1, 'family': edit.family,
                      'affected_ids': list(edit.affected_ids), 'reason': edit.reason,
                      'overall': [key(before)[3], key(after)[3]], 'hard_defects': [-key(before)[2], -key(after)[2]]})
    steps.append({'refine_search': {'mode': 'beam_lns', 'stop_reason': stop or 'max_depth', 'depth': depth + 1,
                                        'evaluated': evaluated, 'visited': len(visited), 'photo_mode': ctx.photo_mode,
                                        'stats': stats.to_dict()}})
    return incumbent, steps
