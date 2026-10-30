"""Floor plans: one layout per canvas shape, sized by role-weighted shrinking.

The idea comes from Gaurav's engine; this is a fresh implementation on our analysis.
Our templates fit every block at one shared scale, so when room runs out the headline
shrinks exactly as much as the fine print. Here each block's scale is

    s_top * k ** speed(role)        (k in (0, 1], found by binary search)

so a single shrink factor `k` drives everything, but badges, legal copy and secondary
images give up room first and the headline, logo and call to action last.

Blocks come from our own analysis, so a text keeps its plate, a phrase stays one unit and
a button keeps its label. Every plan returns an `engine.layout.Placement`; the pipeline
serialises, repairs, checks and ranks it exactly like its own templates.

  tall   (w/h < 0.8)    one column in reading order
  square (0.8 - 1.25)   the same column, plus image-beside-copy when there is a hero
  wide   (1.25 - 3)     hero on its source side, copy column beside it
  banner (>= 3)         one row: logo | copy | hero | call to action
"""
from __future__ import annotations

import math

from engine.blocks import RigidBlock
from engine.canonical import Box
from engine.layout import (ColumnFit, Env, Placement, _gap_units, fit_single, place_backdrop, place_column,
                           place_corners)

# Shrink speeds (exponents on k). 1 shrinks slowest.
SPEED = {'headline': 1.0, 'logo': 1.0, 'cta': 1.0, 'hero': 1.3, 'image': 1.3, 'group': 1.3,
         'subhead': 1.6, 'body': 1.6, 'badges': 2.5, 'badge': 2.5, 'legal': 2.5, 'accent': 2.5, 'sticker': 2.5}
DEFAULT_SPEED = 1.6
TALL_MAX, SQUARE_MAX, WIDE_MAX = 0.8, 1.25, 3.0


def speed(block) -> float:
    return SPEED.get(block.role, DEFAULT_SPEED)


def fit_weighted(env: Env, blocks, rect: Box, s_top: float | None = None, gap_scale: float = 1.0) -> ColumnFit:
    """Largest shrink factor k at which the blocks, each at s_top * k ** speed, stack in `rect`."""
    s_top = s_top or env.s_hi
    units = _gap_units(blocks, env.A.sh)

    def attempt(k: float, ctx) -> ColumnFit:
        laid = [b.layout(s_top * k ** speed(b), rect.w, ctx) for b in blocks]
        gaps = [max(3.0, u * s_top * k * gap_scale) for u in units]
        total = sum(l.h for l in laid) + sum(gaps)
        ok = total <= rect.h + 0.5 and all(l.w <= rect.w + 0.5 for l in laid)
        return ColumnFit(s_top * k, laid, gaps, total, ok, not ctx.floors)

    if not blocks:
        return ColumnFit(0.0, [], [], 0.0, True, False)
    for floors in (True, False):  # readability floors are relaxed only when nothing fits
        ctx = env.ctx(floors)
        best = attempt(1.0, ctx)
        if best.feasible:
            return best
        lo, hi = 1e-3, 1.0
        low = attempt(lo, ctx)
        if not low.feasible:
            continue
        best = low
        for _ in range(16):
            mid = (lo + hi) / 2
            r = attempt(mid, ctx)
            lo, hi, best = (mid, hi, r) if r.feasible else (lo, mid, best)
        return best
    return attempt(1e-3, env.ctx(False))


def _hero(A):
    """The main image block, if the design has one."""
    images = [b for b in A.blocks if b.role in ('hero', 'image')]
    return max(images, key=lambda b: b.src_box.w * b.src_box.h) if images else None


def _inside(inner: Box, outer: Box) -> float:
    return inner.intersection_area(outer) / max(1e-6, inner.w * inner.h)


def _gap(a: Box, b: Box) -> float:
    dx = max(0.0, max(a.x, b.x) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y, b.y) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


def _merge(blocks, block_id: str, lead=None):
    """One rigid unit from several blocks, with the role of `lead` (default: the most important)."""
    lead = lead or max(blocks, key=lambda b: b.importance)
    members = [m for b in blocks for m in b.members]
    return RigidBlock(members, block_id, lead.role, lead.label, lead.importance)


def units(A, scene: bool = True):
    """Layout units: our blocks, with the relations a floor plan must not break made rigid.

    * With `scene`, text and accents printed on the hero photo travel with it (scene unit),
      so copy never leaves the picture it was legible on. Without it, the copy is laid out
      beside the photo: the variant for canvases too small for the whole scene.
    * A small accent (a rule, an icon, a price tag) joins the block it touches or nearly
      touches, so separators stay under their line and icons beside their label.
    Returns (units in reading order, accents that touch nothing, for the corners).
    """
    near = 0.02 * min(A.sw, A.sh)
    hero = _hero(A)
    groups = {id(b): [b] for b in A.blocks}
    owner = {id(b): id(b) for b in A.blocks}
    if hero is not None and scene:
        for b in A.blocks:
            if b is not hero and _inside(b.src_box, hero.src_box) >= 0.8:
                groups[id(hero)] += groups.pop(id(b))
                owner[id(b)] = id(hero)
    loose = []
    for a in A.accents:
        best = min(A.blocks, key=lambda b: _gap(a.src_box, b.src_box), default=None)
        on_hero = scene and hero is not None and _inside(a.src_box, hero.src_box) >= 0.8
        if best is not None and (_gap(a.src_box, best.src_box) <= near or on_hero):
            target = id(hero) if on_hero else owner[id(best)]
            groups[target].append(a)
        else:
            loose.append(a)
    out = []
    for b in A.blocks:
        g = groups.get(id(b))
        if g is None:
            continue
        # A scene keeps the photo's role, so plans still place it as the hero.
        out.append(g[0] if len(g) == 1 else _merge(g, 'unit_' + g[0].id, hero if b is hero else None))
    return out, loose


def _role_unit(us, role):
    return next((u for u in us if u.role == role), None)


def _start(env: Env, template: str, name: str, description: str, scene: bool = True) -> Placement:
    if not scene:
        template, name = template + '_free', name + ' · copy beside the photo'
    P = Placement('floorplan_' + template, 'Floor plan · ' + name, description)
    place_backdrop(env, P)
    return P


def plan_column(env: Env, scene: bool = True) -> Placement | None:
    A = env.A
    us, loose = units(A, scene)
    if not us:
        return None
    P = _start(env, 'column', 'column', 'Every block in reading order; minor items shrink before the headline.', scene)
    m = env.margin
    rect = Box(m, m, env.cw - 2 * m, env.ch - 2 * m)
    fit = fit_weighted(env, us, rect)
    place_column(P, fit, us, rect, 'center')
    place_corners(env, P, A.stickers + loose)
    return P


def plan_split(env: Env, scene: bool = True) -> Placement | None:
    A = env.A
    us, loose = units(A, scene)
    hero = next((u for u in us if u.role in ('hero', 'image')), None)
    if hero is None or len(us) < 2:
        return None
    side = 'right' if hero.src_box.cx > A.sw / 2 else 'left'  # keep the hero on its source side
    P = _start(env, 'split', f'image {side}', 'Hero on its original side; the copy column shrinks minor items first.', scene)
    m, cw, ch = env.margin, env.cw, env.ch
    gutter = max(m, 0.03 * cw)
    inner_w, H = cw - 2 * m - gutter, ch - 2 * m
    hb = hero.src_box
    hero_w = min(max(hb.w / hb.h * H, 0.3 * inner_w), 0.5 * inner_w)
    laid, _ = fit_single(env, hero, hero_w, H, max(env.s_hi, 2.0))
    text_w = inner_w - laid.w
    hx, tx = (m, m + laid.w + gutter) if side == 'left' else (m + text_w + gutter, m)
    patches, boxes = laid.place(hx, m + (H - laid.h) / 2)
    P.add(patches, boxes, hero.id)
    rest = [u for u in us if u is not hero]
    rect = Box(tx, m, text_w, H)
    place_column(P, fit_weighted(env, rest, rect, gap_scale=1.2), rest, rect, 'center')
    place_corners(env, P, A.stickers + loose)
    return P


def plan_row(env: Env, scene: bool = True) -> Placement | None:
    """Banner: logo | copy | hero | call to action, every slot vertically centred."""
    A = env.A
    us, loose = units(A, scene)
    hero = next((u for u in us if u.role in ('hero', 'image')), None)
    logo, cta = _role_unit(us, 'logo'), _role_unit(us, 'cta')
    copy = [u for u in us if u not in (hero, logo, cta)]
    if not copy:
        return None
    P = _start(env, 'row', 'row', 'One row; the copy column gets the room left by logo, hero and button.', scene)
    m, cw, ch = env.margin, env.cw, env.ch
    W, H = cw - 2 * m, ch - 2 * m
    gutter = max(m, 0.02 * cw)
    side_slots = []  # (block, laid) in left-to-right order around the copy
    for block, w_cap, h_cap in ((logo, 0.15, 0.7), (hero, 0.38, 1.0), (cta, 0.22, 0.6)):
        if block is not None:
            laid, _ = fit_single(env, block, w_cap * W, h_cap * H, max(env.s_hi, 2.0))
            side_slots.append((block, laid))
    used = sum(l.w for _, l in side_slots) + gutter * len(side_slots)
    if W - used < 0.3 * W:
        return None
    hero_right = hero is None or hero.src_box.cx > A.sw / 2
    left = [s for s in side_slots if s[0] is logo] + ([s for s in side_slots if s[0] is hero] if not hero_right else [])
    right = ([s for s in side_slots if s[0] is hero] if hero_right else []) + [s for s in side_slots if s[0] is cta]
    x = m
    for block, laid in left:
        patches, boxes = laid.place(x, m + (H - laid.h) / 2)
        P.add(patches, boxes, block.id)
        x += laid.w + gutter
    copy_rect = Box(x, m, W - used, H)
    fit = fit_weighted(env, copy, copy_rect)
    place_column(P, fit, copy, copy_rect, 'center', align='left')
    x = copy_rect.x + copy_rect.w + gutter
    for block, laid in right:
        patches, boxes = laid.place(x, m + (H - laid.h) / 2)
        P.add(patches, boxes, block.id)
        x += laid.w + gutter
    place_corners(env, P, A.stickers + loose)
    return P


def readable(P: Placement, A, cw: float, ch: float) -> bool:
    """Every text stays readable: at least 8 px (2% of the short side on big canvases), or its
    source size if that was smaller. Rigid units can shrink text far below the pipeline's
    floors, so a plan that does is dropped rather than left to the ranking."""
    floor = max(8.0, 0.02 * min(cw, ch))
    for block in A.all_blocks():
        for m in block.members:
            patch = P.patches.get(m['id'], {})
            if m['type'] == 'text' and (m.get('content') or '').strip() and 'font_size' in patch:
                if patch['font_size'] < min(float(m['font_size']), floor):
                    return False
    return True


def plans(A, cw: float, ch: float, emphasis: str = 'balanced') -> list[Placement]:
    """The readable floor plans for this canvas shape (plus the neighbouring shape near a
    boundary), each with copy on the photo and, when that differs, copy beside it."""
    env = Env(A, float(cw), float(ch), emphasis)
    ratio = cw / ch
    makers = []
    if ratio < SQUARE_MAX:
        makers.append(plan_column)
    if TALL_MAX <= ratio < WIDE_MAX * 1.4:
        makers.append(plan_split)
    if ratio >= WIDE_MAX * 0.8:
        makers.append(plan_row)
    with_scene = [len(u.ids) for u in units(A, True)[0]]
    variants = (True, False) if with_scene != [len(u.ids) for u in units(A, False)[0]] else (True,)
    out = []
    for make in makers:
        for scene in variants:
            try:
                p = make(env, scene)
            except (ValueError, ZeroDivisionError, ArithmeticError):
                p = None
            if p is not None and readable(p, A, cw, ch):
                out.append(p)
    return out
