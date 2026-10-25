"""Arrangement proposals and geometry solving.

Each template turns the analysed blocks into one candidate layout for the
target canvas. Templates differ in *structure* (stacked, split, banner row,
proportional); the column/row solvers then search for the largest uniform
scale that fits, with readability floors relaxed only when nothing fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .analysis import Analysis
from .blocks import FLOORS, Block, Ctx, Laid, Patches
from .canonical import Box, element_box, rotated_extent, backdrop_patch

EMPHASIS_BOOST = {
    "balanced": {},
    "headline": {"headline": 1.3, "subhead": 1.1},
    "product": {"hero": 1.35, "image": 1.2},
}


@dataclass
class Placement:
    template: str
    name: str
    description: str
    patches: Patches = field(default_factory=dict)
    boxes: dict[str, Box] = field(default_factory=dict)          # top-level block id -> visual box
    overlay: list[dict[str, Any]] = field(default_factory=list)  # every box, for the viewer
    relaxed_floors: bool = False

    def add(self, patches: Patches, boxes: list[tuple[str, Box]], top_id: str | None = None):
        self.patches.update(patches)
        for bid, b in boxes:
            self.overlay.append({"id": bid, **b.to_dict()})
        if top_id is not None and boxes:
            self.boxes[top_id] = boxes[-1][1]


@dataclass
class Env:
    A: Analysis
    cw: float
    ch: float
    emphasis: str = "balanced"
    scale_bias: float = 1.0

    @property
    def margin(self) -> float:
        return max(5.0, round(0.05 * min(self.cw, self.ch)))

    @property
    def s_hi(self) -> float:
        return self.scale_bias * 1.25 * max(1.0, min(self.cw / self.A.sw, self.ch / self.A.sh))

    def ctx(self, floors: bool = True) -> Ctx:
        return Ctx(self.cw, self.ch, floors=floors, boost=EMPHASIS_BOOST.get(self.emphasis, {}))


# ---------------------------------------------------------------------------
# Fixed layers: background, decoration, corner-anchored stickers/accents
# ---------------------------------------------------------------------------


def place_backdrop(env: Env, P: Placement) -> None:
    A, cw, ch = env.A, env.cw, env.ch
    for e in A.background:
        P.patches[e["id"]] = backdrop_patch(e,cw,ch)
    s = min(cw / A.sw, ch / A.sh)
    for e in A.decorations:
        b = element_box(e)
        w, h = b.w * s, b.h * s
        if w > cw or h > ch:
            k = min(cw / w, ch / h)
            w, h = w * k, h * k
        cx, cy = b.cx / A.sw * cw, b.cy / A.sh * ch
        x = min(max(0.0, cx - w / 2), cw - w)
        y = min(max(0.0, cy - h / 2), ch - h)
        P.patches[e["id"]] = {"x": x, "y": y, "width": w, "height": h}
        P.overlay.append({"id": e["id"], "x": x, "y": y, "w": w, "h": h})


def place_corner(env: Env, P: Placement, block: Block) -> None:
    """Anchor a block to a corner - the source corner it was nearest to,
    unless another corner collides less with what is already placed."""
    A, cw, ch, m = env.A, env.cw, env.ch, env.margin
    b = block.src_box
    src_right, src_bottom = b.cx > A.sw / 2, b.cy > A.sh / 2
    s = min(cw / A.sw, ch / A.sh)
    s = max(s, 0.16 * min(cw, ch) / max(1.0, b.h))      # keep it legible
    s = min(s, 0.3 * cw / b.w, 0.3 * ch / b.h, env.s_hi)
    # Rigid stickers scale as a unit, so their labels ignore Ctx floors. Size
    # them against the copy already placed: keep the source type-size order
    # (a sticker label larger than a badge stays larger) and the role floor,
    # allowing a somewhat larger corner footprint when that is what it takes.
    s_lo, s_up = 0.0, math.inf
    src_fs = {m["id"]: float(m["font_size"]) for blk in A.all_blocks() for m in blk.members if m["type"] == "text"}
    placed = {eid: p["font_size"] for eid, p in P.patches.items() if "font_size" in p and eid in src_fs}
    for m_el in block.members:
        if m_el["type"] != "text" or not m_el.get("content", "").strip():
            continue
        ft = float(m_el["font_size"])
        s_lo = max(s_lo, (FLOORS.get(block.role, 8) + 0.5) / ft)
        for eid, fs in placed.items():
            fp = src_fs[eid]
            if fp < 0.85 * ft:
                s_lo = max(s_lo, (fs + 0.5) / ft)
            elif ft < 0.85 * fp:
                s_up = min(s_up, (fs - 0.5) / ft)
    cap = min(0.45 * cw / b.w, 0.45 * ch / b.h)
    s = min(max(min(s, s_up), min(s_lo, cap)), max(cap, s))
    laid = block.layout(s, max(0.3 * cw, s * b.w + 1), env.ctx())
    ix = (A.sw - b.x1 if src_right else b.x) / A.sw
    iy = (A.sh - b.y1 if src_bottom else b.y) / A.sh
    ix = min(max(ix * cw, m * 0.5), m * 1.5)
    iy = min(max(iy * ch, m * 0.5), m * 1.5)
    best = None
    for right in (src_right, not src_right):
        for bottom in (src_bottom, not src_bottom):
            x = cw - ix - laid.w if right else ix
            y = ch - iy - laid.h if bottom else iy
            x = min(max(0.0, x), cw - laid.w)
            y = min(max(0.0, y), ch - laid.h)
            box = Box(x, y, laid.w, laid.h)
            hit = sum(box.intersection_area(o) for o in P.boxes.values())
            penalty = hit + (right != src_right) * 0.02 * box.area + (bottom != src_bottom) * 0.02 * box.area
            if best is None or penalty < best[0]:
                best = (penalty, x, y)
    patches, boxes = laid.place(best[1], best[2])
    P.add(patches, boxes, block.id)


def place_corners(env: Env, P: Placement, blocks: list[Block]) -> None:
    for b in blocks:
        place_corner(env, P, b)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def _gap_units(blocks: list[Block], sh: float) -> list[float]:
    gaps = []
    for a, b in zip(blocks, blocks[1:]):
        g = b.src_box.y - a.src_box.y1
        if g <= 0:
            g = 0.035 * sh
        gaps.append(min(max(g, 0.015 * sh), 0.07 * sh))
    return gaps


@dataclass
class ColumnFit:
    s: float
    laid: list[Laid]
    gaps: list[float]
    total_h: float
    feasible: bool
    relaxed: bool

    @property
    def width(self) -> float:
        return max((l.w for l in self.laid), default=0.0)

    @property
    def pad_l(self) -> float:
        return max((l.pad[0] for l in self.laid), default=0.0)

    @property
    def pad_r(self) -> float:
        return max((l.pad[2] for l in self.laid), default=0.0)


def fit_column(env: Env, blocks: list[Block], rect: Box, s_hi: float | None = None, gap_scale: float = 1.0) -> ColumnFit:
    """Largest scale s at which the blocks stack inside `rect`."""
    s_hi = s_hi or env.s_hi
    units = _gap_units(blocks, env.A.sh)

    def attempt(s: float, ctx: Ctx) -> ColumnFit:
        laid = [b.layout(s, rect.w, ctx) for b in blocks]
        gaps = [max(3.0, u * s * gap_scale) for u in units]
        total = sum(l.h for l in laid) + sum(gaps)
        ok = total <= rect.h + 0.5 and all(l.w <= rect.w + 0.5 for l in laid)
        return ColumnFit(s, laid, gaps, total, ok, not ctx.floors)

    if not blocks:
        return ColumnFit(0.0, [], [], 0.0, True, False)
    for floors in (True, False):
        ctx = env.ctx(floors)
        best = attempt(s_hi, ctx)
        if best.feasible:
            return best
        lo = 1e-3
        low = attempt(lo, ctx)
        if not low.feasible:
            continue
        hi = s_hi
        best = low
        for _ in range(22):
            mid = (lo + hi) / 2
            r = attempt(mid, ctx)
            if r.feasible:
                lo, best = mid, r
            else:
                hi = mid
        return best
    return attempt(1e-3, env.ctx(False))


def place_column(
    P: Placement,
    fit: ColumnFit,
    blocks: list[Block],
    rect: Box,
    distribute: str = "center",
    src_spacing: list[float] | None = None,
    align: str = "center",
) -> None:
    """distribute: 'center' (compact, centred), 'source' (leftover space
    follows the source's own top/gap/bottom rhythm)."""
    if not blocks:
        return
    leftover = max(0.0, rect.h - fit.total_h)
    n = len(blocks)
    extra = [0.0] * (n + 1)  # before first, between, after last
    if distribute == "source" and src_spacing and len(src_spacing) == n + 1:
        tot = sum(src_spacing) or 1.0
        extra = [leftover * w / tot for w in src_spacing]
    else:
        inner = min(leftover * 0.35, sum(fit.gaps) * 0.6) if n > 1 else 0.0
        for i in range(1, n):
            extra[i] = inner / (n - 1)
        extra[0] = extra[n] = (leftover - inner) / 2
    y = rect.y + extra[0]
    for i, (b, l) in enumerate(zip(blocks, fit.laid)):
        x = rect.x if align == "left" else rect.x + (rect.w - l.w) / 2
        patches, boxes = l.place(x, y)
        P.add(patches, boxes, b.id)
        y += l.h + (fit.gaps[i] + extra[i + 1] if i < n - 1 else 0)
    P.relaxed_floors |= fit.relaxed


def source_spacing(env: Env, blocks: list[Block]) -> list[float]:
    sh = env.A.sh
    if not blocks:
        return []
    sp = [max(0.01 * sh, blocks[0].src_box.y)]
    for a, b in zip(blocks, blocks[1:]):
        sp.append(max(0.01 * sh, b.src_box.y - a.src_box.y1))
    sp.append(max(0.01 * sh, sh - max(b.src_box.y1 for b in blocks)))
    return sp


def fit_single(env: Env, block: Block, max_w: float, max_h: float, s_hi: float | None = None) -> tuple[Laid, bool]:
    fit = fit_column(env, [block], Box(0, 0, max_w, max_h), s_hi)
    return fit.laid[0], fit.relaxed


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def tpl_stack(env: Env, distribute: str) -> Placement:
    name = "Stacked" if distribute == "source" else "Stacked · compact"
    desc = ("Single column in reading order; spacing follows the source rhythm."
            if distribute == "source" else "Single centred column with tight, even spacing.")
    P = Placement(f"stack_{distribute}", name, desc)
    place_backdrop(env, P)
    m = env.margin
    rect = Box(m, m, env.cw - 2 * m, env.ch - 2 * m)
    blocks = env.A.blocks
    fit = fit_column(env, blocks, rect)
    place_column(P, fit, blocks, rect, distribute, source_spacing(env, blocks))
    place_corners(env, P, env.A.stickers + env.A.accents)
    return P


def tpl_split(env: Env, hero_side: str) -> Placement | None:
    A = env.A
    hero = A.block("hero") or A.block("image")
    if hero is None:
        return None
    P = Placement(
        f"split_{hero_side}",
        f"Split · image {hero_side}",
        f"Hero image fills the {hero_side} column; copy, logo and actions stack beside it.",
    )
    place_backdrop(env, P)
    m, cw, ch = env.margin, env.cw, env.ch
    gutter = max(m, 0.03 * cw)
    inner_w, H = cw - 2 * m - gutter, ch - 2 * m
    hb = hero.src_box
    frac = 0.45 if env.emphasis != "product" else 0.55
    hero_w = min(max(hb.w / hb.h * H, 0.3 * inner_w), frac * inner_w)
    hero_s_hi = max(env.s_hi, 2.0)
    laid, _ = fit_single(env, hero, hero_w, H, hero_s_hi)
    text_w = inner_w - laid.w
    if hero_side == "left":
        hx, tx = m, m + laid.w + gutter
    else:
        tx, hx = m, m + text_w + gutter
    patches, boxes = laid.place(hx, m + (H - laid.h) / 2)
    P.add(patches, boxes, hero.id)
    rest = [b for b in A.blocks if b is not hero]
    rect = Box(tx, m, text_w, H)
    fit = fit_column(env, rest, rect, gap_scale=1.2)
    place_column(P, fit, rest, rect, "center")
    place_corners(env, P, A.stickers + A.accents)
    return P


def _row(env: Env, P: Placement, segments: list[tuple[list[Block], float, float]], flex: int) -> None:
    """Lay segments left-to-right. segments: (blocks, max width fraction,
    max height fraction). The `flex` segment (the copy) takes the remaining
    width; non-image blocks elsewhere may not outscale the copy column."""
    m, cw, ch = env.margin, env.cw, env.ch
    W, H = cw - 2 * m, ch - 2 * m
    gutter = max(m, 0.02 * cw)
    flex_seg = segments[flex] if 0 <= flex < len(segments) and segments[flex][0] else None
    segs = [s for s in segments if s[0]]
    fixed = [s for s in segs if s is not flex_seg]
    gutters = gutter * (len(segs) - 1)
    min_flex = 0.34 * W if flex_seg is not None else 0.0

    def fit_fixed(k: float, text_cap: float) -> list[ColumnFit]:
        out = []
        for blocks, wf, hf in fixed:
            rect = Box(0, 0, wf * W * k, hf * H)
            if any(b.role in ("hero", "image", "logo") for b in blocks):
                s_hi = max(env.s_hi, 2.0) if any(b.role == "hero" for b in blocks) else env.s_hi
            else:
                s_hi = text_cap
            out.append(fit_column(env, blocks, rect, s_hi, gap_scale=0.6))
        return out

    def fit_flex(widths: list[float]) -> ColumnFit | None:
        if flex_seg is None:
            return None
        flex_w = max(10.0, W - gutters - sum(widths))
        return fit_column(env, flex_seg[0], Box(0, 0, flex_w, flex_seg[2] * H), gap_scale=0.5)

    # Image padding at the outer ends must stay on-canvas, so reserve it and
    # re-solve if the first pass overhangs.
    for attempt in range(3):
        text_cap = env.s_hi
        for _ in range(2):
            k = 1.0
            fits = fit_fixed(k, text_cap)
            for _ in range(6):
                used = sum(f.width for f in fits)
                if used + gutters + min_flex <= W:
                    break
                k *= max(0.5, (W - gutters - min_flex) / max(1.0, used))
                fits = fit_fixed(k, text_cap)
            widths = [f.width for f in fits]
            flex_fit = fit_flex(widths)
            if flex_fit is None or flex_fit.s >= text_cap * 0.999:
                break
            text_cap = max(flex_fit.s, 1e-3)
    
        order: list[tuple[list[Block], ColumnFit, float, float]] = []
        fi = 0
        for seg in segs:
            if seg is flex_seg:
                order.append((seg[0], flex_fit, max(10.0, W - gutters - sum(widths)), seg[2]))
            else:
                order.append((seg[0], fits[fi], widths[fi], seg[2]))
                fi += 1
        lo = max(m, order[0][1].pad_l)
        hi = cw - max(m, order[-1][1].pad_r)
        reserve = (lo - m) + (cw - m - hi)
        if reserve <= 0.5 or attempt == 2 or W - reserve < W * 0.5:
            break
        W = cw - 2 * m - reserve
        min_flex = 0.34 * W if flex_seg is not None else 0.0

    used = sum(o[2] for o in order)
    # Transparent image margins may overhang neighbours, but never the canvas.
    lo = max(m, order[0][1].pad_l)
    hi = cw - max(m, order[-1][1].pad_r)
    avail = hi - lo
    n = len(order)
    g = min(gutter * 3, gutter + max(0.0, avail - used - gutters) / (n + 1)) if n > 1 else 0.0
    if n > 1 and used + g * (n - 1) > avail:
        g = max(4.0, (avail - used) / (n - 1))
    total = used + g * (n - 1)
    x = lo + max(0.0, avail - total) / 2
    for blocks, fit, w, hf in order:
        seg_h = hf * H
        rect = Box(x, m + (H - seg_h) / 2, w, seg_h)
        place_column(P, fit, blocks, rect, "center")
        x += w + g


def tpl_banner(env: Env, variant: str) -> Placement | None:
    A = env.A
    extreme = env.cw / env.ch >= 5
    logo, hero, cta = A.block("logo"), A.block("hero") or A.block("image"), A.block("cta")
    used = {id(b) for b in (logo, hero, cta) if b}
    copy = [b for b in A.blocks if id(b) not in used]
    badges = [b for b in copy if b.role in ("badges", "badge")]
    one = lambda b: [b] if b else []
    small_h = 0.85 if extreme else 0.55

    if variant == "classic":
        P = Placement("banner_classic", "Banner · classic row",
                      "Logo, product, copy and call to action read left to right.")
        if extreme and badges:
            copy = [b for b in copy if b not in badges]
        segments = [(one(logo), 0.18, small_h), (one(hero), 0.3, 1.0), (copy, 1.0, 1.0),
                    (badges if extreme else [], 0.18, 1.0), (one(cta), 0.22, 0.6 if extreme else 0.5),
                    (A.stickers, 0.16, 0.8)]
        flex, corners = 2, A.accents
    elif variant == "lead":
        if hero is None:
            return None
        P = Placement("banner_lead", "Banner · product lead",
                      "Product anchors the left edge; logo heads the copy column.")
        segments = [(one(hero), 0.3, 1.0), (one(logo) + copy, 1.0, 1.0),
                    (one(cta), 0.22, 0.6 if extreme else 0.5)]
        flex, corners = 1, A.stickers + A.accents
    else:  # "stacked_cta"
        if extreme or cta is None:
            return None
        P = Placement("banner_cta", "Banner · copy with action",
                      "Call to action sits under the copy, leaving room for a larger product.")
        segments = [(one(logo), 0.2, small_h), (one(hero), 0.34, 1.0), (copy + one(cta), 1.0, 1.0)]
        flex, corners = 2, A.stickers + A.accents
    place_backdrop(env, P)
    _row(env, P, segments, flex)
    place_corners(env, P, corners)
    return P


def tpl_scale(env: Env) -> Placement:
    """Uniform scale of the whole composition, centred (letterbox). A safe,
    faithful option when the aspect ratio barely changes."""
    A = env.A
    P = Placement("scale", "Proportional", "Whole composition scaled uniformly and centred.")
    place_backdrop(env, P)
    s = min(env.cw / A.sw, env.ch / A.sh)
    ox, oy = (env.cw - A.sw * s) / 2, (env.ch - A.sh * s) / 2
    ctx = env.ctx(False)
    for b in A.all_blocks():
        laid = b.layout(s, max(1.0, b.src_box.w * s + 1), ctx)
        bx = ox + b.src_box.x * s + (b.src_box.w * s - laid.w) / 2
        by = oy + b.src_box.y * s + (b.src_box.h * s - laid.h) / 2
        bx = min(max(0.0, bx), env.cw - laid.w)
        by = min(max(0.0, by), env.ch - laid.h)
        patches, boxes = laid.place(bx, by)
        P.add(patches, boxes, b.id)
    return P


def tpl_anchored(env: Env) -> Placement:
    """Every block keeps its relative position on the canvas; overlaps that
    appear are pushed apart along the long axis."""
    A, cw, ch, m = env.A, env.cw, env.ch, env.margin
    P = Placement("anchored", "Anchored", "Blocks keep their relative positions; collisions are pushed apart.")
    place_backdrop(env, P)
    vertical = ch / cw >= A.sh / A.sw
    blocks = sorted(A.blocks, key=lambda b: b.src_box.cy if vertical else b.src_box.cx)
    s = min(cw / A.sw, ch / A.sh)
    ctx = env.ctx()
    for _ in range(12):
        placed: list[tuple[Block, Laid, Box]] = []
        for b in blocks:
            sb = b.src_box
            laid = b.layout(s, min(cw - 2 * m, max(sb.w * s, (cw - 2 * m) * 0.6)), ctx)
            cx, cy = sb.cx / A.sw * cw, sb.cy / A.sh * ch
            x = min(max(m, cx - laid.w / 2), cw - m - laid.w)
            y = min(max(m, cy - laid.h / 2), ch - m - laid.h)
            box = Box(x, y, laid.w, laid.h)
            gap = 0.02 * min(cw, ch)
            for _, _, ob in placed:
                if box.intersection_area(Box(ob.x - gap, ob.y - gap, ob.w + 2 * gap, ob.h + 2 * gap)) > 0:
                    if vertical:
                        box = Box(box.x, ob.y1 + gap, box.w, box.h)
                    else:
                        box = Box(ob.x1 + gap, box.y, box.w, box.h)
            placed.append((b, laid, box))
        far = max((pb.y1 if vertical else pb.x1) for _, _, pb in placed) if placed else 0
        limit = (ch if vertical else cw) - m
        if far <= limit + 0.5:
            break
        s *= max(0.6, min(0.95, limit / far))
    for b, laid, box in placed:
        patches, boxes = laid.place(box.x, box.y)
        P.add(patches, boxes, b.id)
    place_corners(env, P, A.stickers + A.accents)
    return P


def _propose(A: Analysis, cw: float, ch: float, emphasis: str = "balanced", bias: float = 1.0) -> list[Placement]:
    env = Env(A, float(cw), float(ch), emphasis, bias)
    rt, rs = cw / ch, A.sw / A.sh
    change = abs(math.log(rt / rs))
    makers: list[Callable[[], Placement | None]] = [lambda: tpl_scale(env)]
    if change < 0.8:
        makers.append(lambda: tpl_anchored(env))
    if rt <= 1.8:
        makers += [lambda: tpl_stack(env, "source"), lambda: tpl_stack(env, "center")]
    if 1.0 <= rt <= 5.0:
        makers += [lambda: tpl_split(env, "left"), lambda: tpl_split(env, "right")]
    if rt >= 2.2:
        makers += [lambda v=v: tpl_banner(env, v) for v in ("classic", "lead", "stacked_cta")]
    out = []
    for make in makers:
        p = make()
        if p is not None:
            out.append(p)
    return out


def propose(A: Analysis, cw: float, ch: float, emphasis: str = "balanced", *,
            portfolio: str | None = None, report: dict | None = None) -> list[Placement]:
    """Bounded structural and scale search, including generous content variants.

    portfolio="legacy" (default; env CHAMELEON_PROPOSAL_PORTFOLIO) is the
    original list. "diverse" drops near-identical geometry (structural
    fingerprint, not template names) and adds a measured, bounded search of
    flowed/hero-limited arrangements kept on a fit/readability/fidelity
    Pareto frontier. `report` (optional dict) receives what was dropped/added.
    """
    portfolio = portfolio_mode(portfolio)
    out=[]
    if len(A.blocks)>=4:
        for cols in (2,3,4):
            if cw/cols>=100:out.append(tpl_columns(Env(A,cw,ch,emphasis),cols))
    for bias in (1.0, .85, 1.3, 1.5):
        for p in _propose(A,cw,ch,emphasis,bias):
            if bias != 1.0:
                p.template += f"_scale{bias}"
                p.name += f" · {bias:g} scale"
            out.append(p)
    out = out[:36]
    if portfolio == "legacy":
        return out
    from .providers.templates import diverse_portfolio
    return diverse_portfolio(A, cw, ch, emphasis, out, report if report is not None else {})


PORTFOLIOS = ("legacy", "diverse")


def portfolio_mode(value: str | None = None) -> str:
    """Explicit keyword first, then CHAMELEON_PROPOSAL_PORTFOLIO; unknown -> legacy."""
    import os
    mode = value if value is not None else os.environ.get("CHAMELEON_PROPOSAL_PORTFOLIO", "legacy")
    return mode if mode in PORTFOLIOS else "legacy"


def tpl_columns(env: Env, columns: int) -> Placement:
    """Flow independent units through columns, rather than one cramped row."""
    A,cw,ch,m=env.A,env.cw,env.ch,env.margin
    P=Placement(f'columns_{columns}',f'{columns} columns','Content flows in reading order through evenly spaced columns.')
    place_backdrop(env,P)
    blocks=A.blocks;gap=.03*min(cw,ch);width=(cw-2*m-gap*(columns-1))/columns
    per=math.ceil(len(blocks)/columns)
    chunks=[blocks[i:i+per] for i in range(0,len(blocks),per)] if per else []
    for i,chunk in enumerate(chunks):
        low,high=0.,max(1.,min(cw/A.sw,ch/A.sh))*env.scale_bias*1.5
        for _ in range(14):
            s=(low+high)/2;laid=[b.layout(s,width,env.ctx(False)) for b in chunk]
            if sum(l.h for l in laid)+gap*max(0,len(laid)-1)<=ch-2*m and all(l.w<=width+.01 for l in laid):low=s
            else:high=s
        y=m
        for b in chunk:
            l=b.layout(low,width,env.ctx(False));patches,boxes=l.place(m+i*(width+gap),y);P.add(patches,boxes,b.id);y+=l.h+gap
    place_corners(env,P,A.stickers+A.accents)
    return P


# ---------------------------------------------------------------------------
# Measured templates for the "diverse" portfolio (bounded parameter search)
# ---------------------------------------------------------------------------

VISUAL_ROLES = ("hero", "image", "logo")


def _partition(heights: list[float], k: int, gap: float) -> list[int]:
    """Contiguous reading-order split into k runs minimising the tallest run."""
    n = len(heights)
    k = max(1, min(k, n))
    pre = [0.0]
    for h in heights:
        pre.append(pre[-1] + h)
    cost = lambda i, j: pre[j] - pre[i] + gap * max(0, j - i - 1)
    INF = math.inf
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for g in range(1, k + 1):
        for j in range(g, n + 1):
            for i in range(g - 1, j):
                v = max(best[g - 1][i], cost(i, j))
                if v < best[g][j] - 1e-9:
                    best[g][j], cut[g][j] = v, i
    bounds, j = [], n
    for g in range(k, 0, -1):
        bounds.append(j)
        j = cut[g][j]
    return sorted(bounds)


def tpl_flow(env: Env, cols: int, visual_side: str, align: str = "center") -> Placement | None:
    """Visuals in a side column; copy flows through `cols` reading-order columns
    sharing one measured scale, so hierarchy is kept across columns."""
    A, cw, ch, m = env.A, env.cw, env.ch, env.margin
    visuals = [b for b in A.blocks if b.role in VISUAL_ROLES]
    copy = [b for b in A.blocks if b not in visuals]
    if not copy or cols > len(copy):
        return None
    # Copy printed on a panel belongs with that panel; flowing it apart strands it.
    if any(p.get("children") for p in A.panels):
        return None
    P = Placement(f"flow_{cols}col_{visual_side}_{align}", f"Flow · {cols} column{'s' if cols > 1 else ''}",
                  f"Copy flows in reading order through {cols} measured column(s); images sit on the {visual_side}.")
    place_backdrop(env, P)
    W, H = cw - 2 * m, ch - 2 * m
    gutter = max(m, 0.025 * cw)
    # Visuals: one strip, each sized to the full height but at most 30% (hero) / 16% (others) of the width.
    vis_laid, used = [], 0.0
    for b in sorted(visuals, key=lambda b: b.src_box.x):
        frac = (0.30 if b.role == "hero" else 0.16) * (1.25 if env.emphasis == "product" else 1.0)
        laid, _ = fit_single(env, b, frac * W, H, max(env.s_hi, 2.0))
        vis_laid.append((b, laid))
        used += laid.w
    used += gutter * len(vis_laid)
    copy_w = W - used
    if copy_w < 0.35 * W:
        return None
    col_gap = max(m, 0.03 * cw)
    col_w = (copy_w - col_gap * (cols - 1)) / cols
    # Balance columns by measured heights at the uniform reference scale.
    s0 = min(cw / A.sw, ch / A.sh)
    probe = [b.layout(max(s0, 1e-3), col_w, env.ctx()).h for b in copy]
    bounds = _partition(probe, cols, 0.02 * ch)
    runs, start = [], 0
    for end in bounds:
        runs.append(copy[start:end])
        start = end
    runs = [r for r in runs if r]
    rects = []
    x0 = m + (used if visual_side == "left" else 0.0)
    for i, _ in enumerate(runs):
        rects.append(Box(x0 + i * (col_w + col_gap), m, col_w, H))
    fits = [fit_column(env, r, rect, gap_scale=0.6) for r, rect in zip(runs, rects)]
    s_common = min(f.s for f in fits)
    fits = [fit_column(env, r, rect, s_hi=max(1e-3, s_common), gap_scale=0.6) for r, rect in zip(runs, rects)]
    # Tighten: centre the used copy width when the columns do not need all of it.
    widths = [f.width for f in fits]
    total = sum(widths) + col_gap * (len(runs) - 1)
    slack = max(0.0, copy_w - total)
    x = x0 + slack / 2
    for r, f, w in zip(runs, fits, widths):
        rect = Box(x, m, w, H)
        place_column(P, f, r, rect, "center", align=align)
        x += w + col_gap
    vx = m if visual_side == "left" else m + copy_w + gutter
    if visual_side == "left":
        vx = m
    for b, laid in vis_laid:
        patches, boxes = laid.place(vx, m + (H - laid.h) / 2)
        P.add(patches, boxes, b.id)
        vx += laid.w + gutter
    place_corners(env, P, A.stickers + A.accents)
    return P


def tpl_stack_hero(env: Env, hero_frac: float) -> Placement | None:
    """Tall formats: the hero may take at most `hero_frac` of the height so the
    copy column gets a measured scale of its own instead of sharing the hero's."""
    A, cw, ch, m = env.A, env.cw, env.ch, env.margin
    hero = A.block("hero") or A.block("image")
    if hero is None:
        return None
    rest = [b for b in A.blocks if b is not hero]
    if not rest or any(p.get("children") for p in A.panels):
        return None
    P = Placement(f"stack_hero{int(round(hero_frac * 100))}", "Stacked · measured hero",
                  f"Hero limited to {hero_frac:.0%} of the height; copy keeps its reading order below/above it.")
    place_backdrop(env, P)
    W, H = cw - 2 * m, ch - 2 * m
    gap = max(m, 0.03 * ch)
    laid, _ = fit_single(env, hero, W, hero_frac * H, max(env.s_hi, 2.0))
    before = sum(1 for b in rest if b.src_box.cy < hero.src_box.cy)
    hero_top = before <= len(rest) / 2
    copy_h = H - laid.h - gap
    if copy_h <= 0.3 * H:
        return None
    rect = Box(m, m + laid.h + gap, W, copy_h) if hero_top else Box(m, m, W, copy_h)
    fit = fit_column(env, rest, rect)
    place_column(P, fit, rest, rect, "center")
    # Hero hugs the copy rather than the canvas edge when the copy is compact.
    used = fit.total_h
    if hero_top:
        y = max(m, rect.y + (rect.h - used) / 2 - gap - laid.h) if used < rect.h else m
    else:
        y = min(ch - m - laid.h, rect.y + (rect.h + used) / 2 + gap) if used < rect.h else ch - m - laid.h
    patches, boxes = laid.place(m + (W - laid.w) / 2, y)
    P.add(patches, boxes, hero.id)
    place_corners(env, P, A.stickers + A.accents)
    return P
