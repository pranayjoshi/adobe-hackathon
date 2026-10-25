"""Group policies: membership separated from layout behaviour.

A GroupPolicy says *which* original elements belong together (membership,
hierarchy, authority, provenance) and *how* the unit may change (mode, allowed
axes, alignment, padding bounds). It never edits the design: original
`group_id`s and every style field stay untouched; the policy lives beside the
design as analysis metadata.

Hierarchy is built from explicit source groups first (high authority), then
inferred label/shape and repeated-row relations, then inferred lockups
(overlapping display type, tightly stacked display lines, accents annotating
copy, logo + wordmark), tables and finally support panels. Every leaf has at
most one *layout owner*; cross-cutting visual relations remain graph edges.
Unsupported nesting collapses to a rigid subtree.

`enumerate_configs` returns a small Pareto set (default cap 8) of measured
configurations for one policy at a target, using the renderer-accurate
typography helpers and visible (alpha) asset bounds. A locked child either keeps
its exact rectangle or the configuration is explicitly marked infeasible.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping

from .canonical import Box, element_box, rotated_extent, union_all, visual_fraction
from .typography import measure, tight_width

POLICY_VERSION = "group-policy/1"
GROUP_POLICY_VERSION = POLICY_VERSION
MODES = ("rigid", "reflowable", "support")
AUTHORITIES = ("explicit", "inferred", "heuristic")
LOCK_TOLERANCE = 1e-6  # A child lock is exact, never a tolerated parent drift.


@dataclass(frozen=True)
class GroupPolicy:
    group_id: str
    member_ids: tuple[str, ...]          # every leaf element in this subtree
    parent: str | None
    authority: str                       # explicit | inferred | heuristic
    mode: str                            # rigid | reflowable | support
    axes: tuple[str, ...]                # allowed arrangement axes; () = fixed relative geometry
    source_order: tuple[str, ...]        # child units (policy ids or leaf ids) in reading order
    alignment: str                       # left | center | right | source
    padding_bounds: tuple[float, float] | None  # (min, max) padding, source px
    provenance: str
    kind: str = "group"                  # button | badge_row | text_lockup | display_stack | accent_lockup | logo_lockup | table | panel | explicit | sticker
    children: tuple[str, ...] = ()       # child policy ids
    support_id: str | None = None        # backplate element of a support policy
    confidence: float = 1.0

    def owned_ids(self, policies: Mapping[str, "GroupPolicy"]) -> tuple[str, ...]:
        nested = {i for c in self.children if c in policies for i in policies[c].member_ids}
        return tuple(i for i in self.member_ids if i not in nested)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["version"] = POLICY_VERSION
        d['editable_modes']=(['support','rigid'] if self.kind=='card' else
                             ['reflowable','rigid'] if self.kind in ('button','display_stack','badge_row','card_grid') else ['rigid'])
        return d


@dataclass(frozen=True)
class GroupConfig:
    id: str
    group_id: str
    width: float
    height: float
    child_patches: Mapping[str, Mapping[str, float]]  # offsets from the group origin (or absolute when anchored)
    source_order: tuple[str, ...]
    penalties: Mapping[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.penalties.get("infeasible")

    @property
    def anchored(self) -> bool:
        return bool(self.penalties.get("anchored"))

    def place(self, x: float, y: float) -> dict[str, dict[str, float]]:
        """Absolute geometry patches. Anchored (lock-pinned) configs ignore x, y."""
        if not self.feasible:
            raise ValueError(f"configuration {self.id} is infeasible")
        if self.anchored:
            x, y = self.penalties.get("origin_x", 0.0), self.penalties.get("origin_y", 0.0)
        out = {}
        for eid, p in self.child_patches.items():
            q = dict(p)
            q["x"] = p["x"] + x
            q["y"] = p["y"] + y
            out[eid] = q
        return out


class PolicyError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(policies: Iterable[GroupPolicy], element_ids: Iterable[str]) -> list[GroupPolicy]:
    """Check acyclicity, parent/child consistency, single layout ownership of
    every leaf and that photographs are never flexible supports. Raises
    PolicyError; callers fall back to legacy grouping on failure."""
    pols = list(policies)
    by = {p.group_id: p for p in pols}
    if len(by) != len(pols):
        raise PolicyError("duplicate policy id")
    known = set(element_ids)
    owner: dict[str, str] = {}
    for p in pols:
        if p.mode not in MODES or p.authority not in AUTHORITIES:
            raise PolicyError(f"{p.group_id}: invalid mode/authority")
        if not p.member_ids or not set(p.member_ids) <= known:
            raise PolicyError(f"{p.group_id}: unknown or empty members")
        if p.parent is not None and (p.parent not in by or p.group_id not in by[p.parent].children):
            raise PolicyError(f"{p.group_id}: inconsistent parent")
        for c in p.children:
            if c not in by or by[c].parent != p.group_id:
                raise PolicyError(f"{p.group_id}: inconsistent child {c}")
            if not set(by[c].member_ids) <= set(p.member_ids):
                raise PolicyError(f"{p.group_id}: child {c} not inside membership")
        for i in p.owned_ids(by):
            if i in owner:
                raise PolicyError(f"leaf {i} owned by {owner[i]} and {p.group_id}")
            owner[i] = p.group_id
    for p in pols:  # acyclic parent chains
        seen, cur = set(), p
        while cur.parent is not None:
            if cur.group_id in seen:
                raise PolicyError(f"cycle at {p.group_id}")
            seen.add(cur.group_id)
            cur = by[cur.parent]
    return pols


def roots(policies: Iterable[GroupPolicy]) -> list[GroupPolicy]:
    return [p for p in policies if p.parent is None]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _fs(e: dict[str, Any]) -> float:
    return float(e.get("font_size") or 0)


def _gap(a: Box, b: Box) -> float:
    dx = max(0.0, max(a.x, b.x) - min(a.x1, b.x1))
    dy = max(0.0, max(a.y, b.y) - min(a.y1, b.y1))
    return math.hypot(dx, dy)


def _hoverlap(a: Box, b: Box) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x, b.x)) / max(1e-6, min(a.w, b.w))


def _voverlap(a: Box, b: Box) -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y, b.y)) / max(1e-6, min(a.h, b.h))


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def groups(self):
        out: dict[str, list[str]] = {}
        for i in self.p:
            out.setdefault(self.find(i), []).append(i)
        return [sorted(v) for v in out.values()]


def text_lockups(texts: list[dict[str, Any]], glyph) -> list[list[str]]:
    """Display type with a smaller mark nested on it (%, ®, 'up to', currency):
    glyph bands overlap and sizes differ by >= 1.5x. Equal-size overlapping
    copy is not a lockup (more often an import artefact)."""
    uf = _UF([t["id"] for t in texts])
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            ga, gb = glyph[a["id"]], glyph[b["id"]]
            inter = ga.intersection_area(gb)
            if inter <= 0:
                continue
            big, small = (a, b) if _fs(a) >= _fs(b) else (b, a)
            if _fs(big) >= 1.5 * max(1e-6, _fs(small)) and inter >= 0.5 * glyph[small["id"]].area:
                uf.union(a["id"], b["id"])
    return [g for g in uf.groups() if len(g) > 1]


def display_stacks(units: list[tuple[str, Box, float]], display_fs: float) -> list[list[str]]:
    """Display lines set tighter than ordinary leading: vertical gap below a
    quarter of the smaller size, horizontally overlapping, both display-sized."""
    uf = _UF([u[0] for u in units])
    for i, (a, ba, fa) in enumerate(units):
        for b, bb, fb in units[i + 1:]:
            if min(fa, fb) < display_fs:
                continue
            top, bot = (ba, bb) if ba.y <= bb.y else (bb, ba)
            gap = bot.y - top.y1
            if -0.25 * min(fa, fb) <= gap <= 0.25 * min(fa, fb) and _hoverlap(ba, bb) >= 0.5:
                uf.union(a, b)
    return [g for g in uf.groups() if len(g) > 1]


def table_cells(texts: list[dict[str, Any]], glyph, sw: float, sh: float) -> list[list[str]]:
    """A ruled table / price list / form: >= 2 left-aligned columns of small
    copy whose rows line up across columns (>= 2 shared rows). Linearising it
    would separate each value from its label."""
    if len(texts) < 4:
        return []
    tol_x, tol_y = 0.012 * sw, 0.008 * sh
    cols: list[list[dict[str, Any]]] = []
    for t in sorted(texts, key=lambda t: glyph[t["id"]].x):
        g = glyph[t["id"]]
        for c in cols:
            if abs(glyph[c[0]["id"]].x - g.x) <= tol_x:
                c.append(t)
                break
        else:
            cols.append([t])
    cols = [c for c in cols if len(c) >= 2]
    best: list[str] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rows = [(x, y) for x in a for y in b if abs(glyph[x["id"]].y - glyph[y["id"]].y) <= tol_y]
            if len(rows) < 2:
                continue
            top = min(glyph[x["id"]].y for x, _ in rows) - tol_y
            bottom = max(glyph[x["id"]].y for x, _ in rows) + tol_y
            ca = [t for t in a if top <= glyph[t["id"]].y <= bottom]
            cb = [t for t in b if top <= glyph[t["id"]].y <= bottom]
            members = {t["id"] for t in ca} | {t["id"] for t in cb}
            ga, gb = union_all(glyph[t["id"]] for t in ca), union_all(glyph[t["id"]] for t in cb)
            # Columns must be neighbours (no unrelated copy far away) and overlap vertically.
            if _voverlap(ga, gb) < 0.5 or max(ga.x, gb.x) - min(ga.x1, gb.x1) > 0.35 * sw:
                continue
            if len(members) > len(best):
                best = sorted(members)
    return [best] if best else []


@dataclass
class InferenceContext:
    source: dict[str, Any]
    glyph: dict[str, Box]
    kinds: dict[str, str]            # image id -> pixel kind (photo/graphic/linework/flat/missing)
    roles: dict[str, str]            # provisional roles for images (logo/hero/...)
    loose_text: list[dict[str, Any]]
    loose_images: list[dict[str, Any]]
    features: frozenset = frozenset({"lockups", "stacks", "accents", "logos", "tables"})


def infer_lockups(ctx: InferenceContext) -> list[GroupPolicy]:
    """Inferred rigid/reflowable lockups among loose elements (no explicit group)."""
    src = ctx.source
    sw, sh = float(src["canvas"]["width"]), float(src["canvas"]["height"])
    texts = [t for t in ctx.loose_text if str(t.get("content", "")).strip() and abs(float(t.get("rotation") or 0)) <= 0.5]
    by = {e["id"]: e for e in src["elements"]}
    pols: list[GroupPolicy] = []
    owned: set[str] = set()
    unit_of: dict[str, str] = {}

    def add(p: GroupPolicy):
        pols.append(p)
        for i in p.member_ids:
            unit_of[i] = p.group_id
        owned.update(p.member_ids)

    def order(ids):
        return tuple(sorted(ids, key=lambda i: (round(ctx.glyph.get(i, element_box(by[i])).y / (0.02 * sh)), ctx.glyph.get(i, element_box(by[i])).x)))

    n = 0
    if "lockups" in ctx.features:
        for g in text_lockups(texts, ctx.glyph):
            n += 1
            add(GroupPolicy(f"lockup_{n}", order(g), None, "inferred", "rigid", (), order(g), "source", None,
                            "inferred:glyph-overlap with >=1.5x size ratio", kind="text_lockup", confidence=0.8))
    # Accents annotating copy: a small graphic/rule/icon next to the text it marks.
    if "accents" in ctx.features:
        area = sw * sh
        for img in ctx.loose_images:
            if ctx.kinds.get(img["id"]) not in ("graphic", "flat", "linework") or abs(float(img.get("rotation") or 0)) > 0.5 \
                    or ctx.roles.get(img["id"]) == "logo":  # a logo is a brand mark, handled below
                continue
            vb = _vis(img)
            if vb.area > 0.015 * area or vb.area <= 0:
                continue
            near = []
            unit = max(1.0, min(vb.w, vb.h))
            for t in texts:
                g = ctx.glyph[t["id"]]
                gap = _gap(vb, g)
                # Proximity in the larger of the two local scales (type size
                # or the accent's own thickness, e.g. an icon beside small copy).
                reach = 1.25 * max(_fs(t), unit)
                # ...and no further than twice the accent's own length: a
                # sprinkled ornament far from copy annotates nothing.
                if gap > reach or gap > 2 * max(vb.w, vb.h):
                    continue
                if _hoverlap(vb, g) > 0 or _voverlap(vb, g) > 0:
                    near.append((gap / reach, gap, t))
            if not near:
                continue
            near.sort(key=lambda x: x[1])  # an accent annotates its nearest copy
            first = near[0][2]
            chosen = [first]
            reach0 = 1.25 * max(_fs(first), unit)
            # A rule spanning several stacked lines annotates all of them, when
            # they are as close to it as the nearest one is allowed to be.
            vertical_rule = vb.h >= 3 * vb.w
            horizontal_rule = vb.w >= 3 * vb.h
            for t in texts:
                if t is first:
                    continue
                g = ctx.glyph[t["id"]]
                spans = (vertical_rule and _voverlap(vb, g) >= 0.9 and g.h <= vb.h + 1) or \
                        (horizontal_rule and _hoverlap(vb, g) >= 0.9 and g.w <= vb.w + 1)
                if spans and _gap(vb, g) <= min(reach0, near[0][1] + 1.25 * max(_fs(t), unit)):
                    chosen.append(t)
            ids = [img["id"]] + [t["id"] for t in chosen]
            # Merge with existing units sharing these texts.
            joined = {unit_of[i] for i in ids if i in unit_of}
            if joined:
                members = set(ids)
                for pid in joined:
                    p = next(p for p in pols if p.group_id == pid)
                    members |= set(p.member_ids)
                    pols.remove(p)
                ids = sorted(members)
            n += 1
            add(GroupPolicy(f"lockup_{n}", order(ids), None, "inferred", "rigid", (), order(ids), "source", None,
                            "inferred:small graphic accent within 1.25em of the copy it annotates", kind="accent_lockup", confidence=0.65))
    # Logo + wordmark.
    if "logos" in ctx.features:
        for img in ctx.loose_images:
            if ctx.roles.get(img["id"]) != "logo" or img["id"] in owned:
                continue
            vb = _vis(img)
            best = None
            for t in texts:
                if t["id"] in owned or len(str(t["content"]).split()) > 3:
                    continue
                g = ctx.glyph[t["id"]]
                gap = _gap(vb, g)
                stacked = _hoverlap(vb, g) >= 0.6 and abs(vb.cx - g.cx) <= 0.25 * max(vb.w, g.w)
                beside = _voverlap(vb, g) >= 0.6
                if gap <= 1.0 * _fs(t) and (stacked or beside) and (best is None or gap < best[0]):
                    best = (gap, t)
            if best:
                ids = [img["id"], best[1]["id"]]
                n += 1
                add(GroupPolicy(f"lockup_{n}", order(ids), None, "inferred", "rigid", (), order(ids), "source", None,
                                "inferred:logo image with a <=3-word wordmark within 1em", kind="logo_lockup", confidence=0.6))
    # Display stacks (reflowable along y or x; nested lockups stay rigid).
    if "stacks" in ctx.features and texts:
        sizes = sorted(_fs(t) for t in texts)
        display = 1.5 * statistics.median(sizes)
        units: list[tuple[str, Box, float]] = []
        unit_members: dict[str, tuple[str, ...]] = {}
        for p in pols:
            if p.kind == "text_lockup":
                ms = [by[i] for i in p.member_ids]
                units.append((p.group_id, union_all(ctx.glyph[i] for i in p.member_ids), max(_fs(m) for m in ms)))
                unit_members[p.group_id] = p.member_ids
        for t in texts:
            if t["id"] not in owned:
                units.append((t["id"], ctx.glyph[t["id"]], _fs(t)))
                unit_members[t["id"]] = (t["id"],)
        for g in display_stacks(units, display):
            members = tuple(i for u in g for i in unit_members[u])
            kids = tuple(u for u in g if u in unit_members and u.startswith("lockup_"))
            n += 1
            sid = f"stack_{n}"
            child_order = tuple(sorted(g, key=lambda u: next(b.y for k, b, _ in units if k == u)))
            pols = [GroupPolicy(p.group_id, p.member_ids, sid, p.authority, p.mode, p.axes, p.source_order, p.alignment,
                                p.padding_bounds, p.provenance, p.kind, p.children, p.support_id, p.confidence)
                    if p.group_id in kids else p for p in pols]
            src_align = "left" if all(abs(next(b.x for k, b, _ in units if k == u) - next(b.x for k, b, _ in units if k == g[0])) <= 0.012 * sw for u in g) else "center"
            add(GroupPolicy(sid, order(members), None, "inferred", "reflowable", ("y", "x"), child_order, src_align, None,
                            "inferred:display lines stacked tighter than 0.25em", kind="display_stack", children=kids, confidence=0.6))
    # Tables over the copy that is still independent.
    if "tables" in ctx.features:
        free = [t for t in texts if t["id"] not in owned]
        for cells in table_cells(free, ctx.glyph, sw, sh):
            region = union_all(ctx.glyph[i] for i in cells)
            # Ruling / linework drawn over the table area belongs to it.
            rules = [img["id"] for img in ctx.loose_images if img["id"] not in owned and ctx.kinds.get(img["id"]) == "linework"
                     and _vis(img).intersection_area(region) >= 0.6 * max(1e-6, _vis(img).area)]
            # Labels directly above / below inside the same columns (headers, totals).
            ids = sorted(set(cells) | set(rules))
            n += 1
            add(GroupPolicy(f"table_{n}", order(ids), None, "heuristic", "rigid", (), order(ids), "left", None,
                            "heuristic:>=2 aligned columns sharing >=2 rows", kind="table", confidence=0.55))
    return pols


def _vis(e: dict[str, Any]) -> Box:
    b = element_box(e)
    if e.get("type") != "image":
        return rotated_extent(e)
    fl, ft, fw, fh = visual_fraction(e)
    return Box(b.x + fl * b.w, b.y + ft * b.h, fw * b.w, fh * b.h)


def policies_from_blocks(source: dict[str, Any], analysis: Any) -> list[GroupPolicy]:
    """Describe the block structure (legacy or hierarchical) as validated policies."""
    from .blocks import ButtonBlock, CardBlock, GridBlock, RigidBlock, RowBlock, StackBlock

    pols: list[GroupPolicy] = []

    def emit(b, parent) -> str | None:
        if isinstance(b, (RowBlock, StackBlock, CardBlock, GridBlock)):
            kids = [k for k in (emit(c, b.id) for c in b.children) if k]
            base = getattr(b, "policy", None)
            if isinstance(b, RowBlock):
                p = GroupPolicy(b.id, tuple(b.ids), parent, "inferred", "reflowable", ("x", "y"), tuple(kids), "center", None,
                                "legacy:RowBlock badges sharing a baseline", kind="badge_row", children=tuple(kids))
            else:
                p = GroupPolicy(b.id, tuple(b.ids), parent, base.authority if base else "inferred", base.mode if base else "reflowable",
                                base.axes if base else ("y", "x"), tuple(c.id for c in b.children), base.alignment if base else "source",
                                base.padding_bounds if base else None, base.provenance if base else type(b).__name__,
                                kind=getattr(b, "kind", "group"), children=tuple(kids),
                                support_id=getattr(b, "panel", {}).get("id") if isinstance(b, CardBlock) else None,
                                confidence=base.confidence if base else 0.5)
            pols.append(p)
            return b.id
        if isinstance(b, ButtonBlock):
            pols.append(GroupPolicy(b.id, tuple(b.ids), parent, _authority(b.members), "reflowable", ("x",), tuple(b.ids), "center",
                                    _pad_bounds(b), "legacy:ButtonBlock", kind="button"))
            return b.id
        if isinstance(b, RigidBlock):
            base = getattr(b, "policy", None)
            pols.append(GroupPolicy(b.id, tuple(b.ids), parent, base.authority if base else _authority(b.members), "rigid", (), tuple(b.ids),
                                    "source", None, base.provenance if base else f"legacy:RigidBlock({b.role})",
                                    kind="sticker" if b.role == "sticker" else getattr(b, "kind", "explicit"),
                                    confidence=base.confidence if base else 1.0))
            return b.id
        return None

    for b in analysis.all_blocks():
        emit(b, None)
    for panel in getattr(analysis, "panels", []):
        pid = f"panel:{panel['id']}"
        if any(p.group_id == pid for p in pols) or panel["id"] not in {e["id"] for e in source["elements"]}:
            continue
        # Legacy panels cross-cut block ownership: own only the backplate; the
        # text it carries is recorded as graph 'contains' edges.
        pols.append(GroupPolicy(pid, (panel["id"],), None, "heuristic", "support", ("x", "y"), tuple(panel.get("children", [])), "source",
                                None, "legacy:panel grows around contained text (children are graph edges)", kind="panel", support_id=panel["id"]))
    try:
        return validate(pols, [e["id"] for e in source["elements"]])
    except PolicyError:
        return []


def _authority(members) -> str:
    return "explicit" if members and all(m.get("group_id") for m in members) and len({m.get("group_id") for m in members}) == 1 else "inferred"


def _pad_bounds(b) -> tuple[float, float]:
    lo = float(b.src_fs) * 0.6
    return (round(lo, 2), round(max(lo, float(b.pad_x), float(b.pad_y)) * 1.5, 2))


# ---------------------------------------------------------------------------
# Measured configurations
# ---------------------------------------------------------------------------


def _shrink(el: dict[str, Any], fs: int, width: float):
    fit = measure(el, fs, width)
    while fit.overflow and fit.font_size > 1:
        fit = measure(el, fit.font_size - 1, width)
    return fit


def _floor(role: str) -> int:
    from .blocks import FLOORS

    return FLOORS.get(role, 8)


def _rigid_patches(members: list[dict[str, Any]], s: float) -> tuple[dict[str, dict[str, float]], Box, float]:
    """Uniform scale about the union origin; text sizes are integers that still
    fit their scaled width (never wider than the scaled box)."""
    U = union_all(rotated_extent(m) for m in members)
    patches: dict[str, dict[str, float]] = {}
    worst = math.inf
    for m in members:
        b = element_box(m)
        p = {"x": (b.x - U.x) * s, "y": (b.y - U.y) * s, "width": b.w * s, "height": b.h * s}
        if m["type"] == "text":
            fit = _shrink(m, max(1, round(_fs(m) * s)), b.w * s)
            while fit.font_size>1 and fit.height>b.h*s+.5:
                fit=measure(m,fit.font_size-1,b.w*s)
            p["font_size"] = fit.font_size
            worst = min(worst, fit.font_size / max(1.0, min(_fs(m), _floor("body"))))
        patches[m["id"]] = p
    return patches, U, (worst if worst != math.inf else 1.0)


def _scales(U: Box, target: Mapping[str, float], sw: float, sh: float) -> list[float]:
    tw, th = float(target["width"]), float(target["height"])
    base = min(tw / sw, th / sh)
    area = math.sqrt(tw * th / (sw * sh))
    fit = min(tw / max(1e-6, U.w), th / max(1e-6, U.h))
    cands = {round(v, 4) for v in (base, area, fit, 0.75 * fit, 0.5 * fit, 0.33 * fit, min(1.0, fit)) if v > 0}
    return sorted(c for c in cands if c <= fit + 1e-9)


def _pareto(configs: list[GroupConfig], cap: int) -> list[GroupConfig]:
    """Keep configurations not dominated on (width, height, -readability)."""
    def key(c):
        return (c.width, c.height, -c.penalties.get("readability", 0.0))

    keep = []
    for c in configs:
        kc = key(c)
        dominated = any(all(x <= y for x, y in zip(key(o), kc)) and key(o) != kc for o in configs if o is not c and o.feasible == c.feasible)
        if not dominated and all(key(o) != kc for o in keep):
            keep.append(c)
    keep.sort(key=lambda c: (not c.feasible, c.width * c.height))
    if len(keep) <= cap:
        return keep
    # Spread the cap across the size range rather than truncating one end.
    if cap == 1:
        return keep[:1]
    idx = sorted({round(i * (len(keep) - 1) / (cap - 1)) for i in range(cap)})
    return [keep[i] for i in idx]


def enumerate_configs(policy: GroupPolicy, design_or_source: dict[str, Any], analysis: Any, target: Mapping[str, float],
                      max_configs: int = 8, *, locks: Mapping[str, Mapping[str, float]] | None = None,
                      policies: Iterable[GroupPolicy] | None = None) -> list[GroupConfig]:
    """Bounded Pareto set of measured configurations for `policy` at `target`.

    Geometry is derived from the immutable source elements (sizes, fonts) via
    renderer-accurate text measurement. Locks (target-canvas rectangles keyed
    by element id) pin children exactly: each configuration is translated to
    satisfy the lock, or carries `penalties['infeasible'] = 1`.
    """
    els = {e["id"]: e for e in design_or_source["elements"]}
    members = [els[i] for i in policy.member_ids if i in els]
    if not members:
        return []
    sw = float(getattr(analysis, "sw", design_or_source["canvas"]["width"]))
    sh = float(getattr(analysis, "sh", design_or_source["canvas"]["height"]))
    by_pol = {p.group_id: p for p in (policies or getattr(analysis, "policies", []) or [])}
    cap = max(1, int(max_configs))
    out: list[GroupConfig] = []
    if policy.mode=='reflowable' and (policy.kind == "button" or (len(members) == 2 and {m["type"] for m in members} == {"shape", "text"})):
        out = _button_configs(policy, members, target, sw, sh)
    elif policy.mode == "reflowable" and policy.children or policy.kind in ("badge_row", "display_stack"):
        pinned_scales=[]
        for member in members:
            if locks and member['id'] in locks:
                rectangle=locks[member['id']]
                ratios=[float(rectangle[key])/float(member[key]) for key in ('width','height') if key in rectangle]
                if ratios and max(ratios)-min(ratios)<=LOCK_TOLERANCE:pinned_scales.append(ratios[0])
        out = _arrangement_configs(policy, members, els, by_pol, target, sw, sh, cap,pinned_scales)
    elif policy.mode == "support":
        out = _support_configs(policy, members, els, target, sw, sh)
    else:
        U = union_all(rotated_extent(m) for m in members)
        for s in _scales(U, target, sw, sh):
            patches, U, read = _rigid_patches(members, s)
            out.append(GroupConfig(f"{policy.group_id}:rigid@{s:.3f}", policy.group_id, U.w * s, U.h * s, patches,
                                   policy.source_order, {"readability": round(read, 4), "scale": s, "axis_change": 0.0}))
    if locks:
        # Pin before pruning. Otherwise a size-spread cap can discard the one
        # scale that exactly satisfies a child lock.
        out = [_apply_locks(c, policy, members, locks) for c in out] + _lock_derived(policy, members, locks, target, sw, sh)
        out = _pareto(out, cap) if any(c.feasible for c in out) else out[:cap]
    else:
        out=_pareto(out,cap)
    return out[:cap]


def _button_configs(policy, members, target, sw, sh) -> list[GroupConfig]:
    shape = next(m for m in members if m["type"] == "shape")
    text = next(m for m in members if m["type"] == "text")
    src_fs = int(_fs(text))
    src_fit = measure(text, src_fs, float(text["width"]))
    pad_x0 = max(src_fs * 1.2, float(shape["width"]) - src_fit.line_width)
    pad_y0 = max(src_fs * 0.6, float(shape["height"]) - src_fit.height)
    base = min(float(target["width"]) / sw, float(target["height"]) / sh)
    sizes = sorted({max(1, round(src_fs * k)) for k in (base, math.sqrt(base), 1.0, 0.75 * base)} | {min(src_fs, max(_floor("cta"), 1))})
    out = []
    for fs in sizes:
        k = fs / max(1, src_fs)
        for pad_name, (px, py) in (("source_padding", (pad_x0 * k, pad_y0 * k)), ("min_padding", (fs * 1.2, fs * 0.6))):
            for lines in (1, 2):
                if lines == 1:
                    fit = measure(text, fs, 1e6)
                else:
                    if src_fit.lines > 1 or len(str(text["content"]).split()) < 2:
                        continue
                    fit = measure(text, fs, measure(text, fs, 1e6).line_width * 0.6)
                    if fit.lines != 2 or fit.overflow:
                        continue
                w, h = fit.line_width + px, fit.height + py
                if w > float(target["width"]) or h > float(target["height"]):
                    continue
                tw = tight_width(text, fit)
                patches = {shape["id"]: {"x": 0.0, "y": 0.0, "width": w, "height": h},
                           text["id"]: {"x": (w - tw) / 2, "y": h / 2 - fit.visual_center, "width": tw,
                                        "height": fit.height, "font_size": fit.font_size}}
                out.append(GroupConfig(f"{policy.group_id}:fs{fs}:{pad_name}:{lines}l", policy.group_id, w, h, patches, policy.source_order,
                                       {"readability": round(fs / max(1, min(src_fs, _floor('cta'))), 4), "padding_change": float(pad_name != "source_padding"),
                                        "wrap": float(lines - 1)}))
    return out


def _unit_layout(uid, els, by_pol, s):
    """Rigid geometry of one child unit (policy or leaf) at uniform scale s."""
    ms = [els[i] for i in (by_pol[uid].member_ids if uid in by_pol else (uid,)) if i in els]
    patches, U, read = _rigid_patches(ms, s)
    return patches, U.w * s, U.h * s, read


def _arrangement_configs(policy, members, els, by_pol, target, sw, sh, cap,extra_scales=()) -> list[GroupConfig]:
    order = [u for u in policy.source_order if u in by_pol or u in els]
    if not order:
        order = [m["id"] for m in members]
    U = union_all(rotated_extent(m) for m in members)
    src_axis = 'source' if policy.kind=='card_grid' else "y" if policy.kind == "display_stack" else "x"
    gaps = []
    boxes = [union_all(rotated_extent(els[i]) for i in (by_pol[u].member_ids if u in by_pol else (u,))) for u in order]
    for a, b in zip(boxes, boxes[1:]):
        gaps.append(max(0.0, (b.y - a.y1) if src_axis == "y" else (b.x - a.x1)))
    gap0 = statistics.median(gaps) if gaps else 0.0
    out = []
    scales=_scales(U,target,sw,sh)+[1.0,*extra_scales,min(float(target["width"]) / max(1e-6, sum(b.w for b in boxes)), float(target["height"]) / max(1e-6, max(b.h for b in boxes)))]
    for s in sorted(set(scales)):
        laid = [_unit_layout(u, els, by_pol, s) for u in order]
        read = min(l[3] for l in laid)
        for axis in policy.axes or (src_axis,):
            patches: dict[str, dict[str, float]] = {}
            gap = max(2.0, gap0 * s) if axis == src_axis else max(2.0, 0.3 * statistics.median(l[2] for l in laid))
            if axis=='source':
                for (p,w,h,_),box in zip(laid,boxes):
                    for eid,q in p.items():patches[eid]={**q,'x':q['x']+(box.x-U.x)*s,'y':q['y']+(box.y-U.y)*s}
                W,H=U.w*s,U.h*s
            elif axis == "y":
                W = max(l[1] for l in laid)
                y = 0.0
                for (p, w, h, _), u in zip(laid, order):
                    x = (W - w) / 2 if policy.alignment == "center" else 0.0
                    for eid, q in p.items():
                        patches[eid] = {**q, "x": q["x"] + x, "y": q["y"] + y}
                    y += h + gap
                H = y - gap
            else:
                H = max(l[2] for l in laid)
                x = 0.0
                for (p, w, h, _), u in zip(laid, order):
                    yy = H - h  # bottom-aligned: display type shares a baseline region
                    for eid, q in p.items():
                        patches[eid] = {**q, "x": q["x"] + x, "y": q["y"] + yy}
                    x += w + gap
                W = x - gap
            if W > float(target["width"]) + 0.5 or H > float(target["height"]) + 0.5:
                continue
            out.append(GroupConfig(f"{policy.group_id}:{axis}@{s:.3f}", policy.group_id, W, H, patches, tuple(order),
                                   {"readability": round(read, 4), "scale": s, "axis_change": float(axis != src_axis), "order_change": 0.0}))
    return out


def _support_configs(policy, members, els, target, sw, sh) -> list[GroupConfig]:
    """Backplate follows its children: children keep relative geometry at s and
    the panel spans their bounds plus scaled source padding (within bounds)."""
    panel = els.get(policy.support_id or "")
    kids = [m for m in members if m["id"] != (policy.support_id or "")]
    if panel is None or not kids:
        return []
    K = union_all(rotated_extent(m) for m in kids)
    P = element_box(panel)
    pads = (K.x - P.x, K.y - P.y, P.x1 - K.x1, P.y1 - K.y1)
    lo, hi = policy.padding_bounds or (0.0, math.inf)
    out = []
    for s in _scales(P, target, sw, sh):
        patches, U, read = _rigid_patches(kids, s)
        l, t, r, b = (min(hi, max(lo, p * s)) if p > 0 else 0.0 for p in pads)
        for q in patches.values():
            q["x"] += l
            q["y"] += t
        W, H = U.w * s + l + r, U.h * s + t + b
        if panel['type'] == 'image':
            # Raster supports retain their original aspect. Grow their common
            # scale until the measured children fit, instead of stretching.
            factor=max(W/max(1e-6,P.w),H/max(1e-6,P.h))
            pw,ph=P.w*factor,P.h*factor
            for patch in patches.values():
                patch['x']+=(pw-W)/2
                patch['y']+=(ph-H)/2
            W,H=pw,ph
        patches[panel["id"]] = {"x": 0.0, "y": 0.0, "width": W, "height": H}
        out.append(GroupConfig(f"{policy.group_id}:support@{s:.3f}", policy.group_id, W, H, patches, policy.source_order,
                               {"readability": round(read, 4), "scale": s}))
    return out


def _matches(p: Mapping[str, float], lock: Mapping[str, float], dims=("width", "height")) -> bool:
    return all(abs(float(p[k]) - float(lock[k])) < LOCK_TOLERANCE for k in dims if k in lock and k in p)


def _apply_locks(c: GroupConfig, policy, members, locks) -> GroupConfig:
    pinned = [i for i in policy.member_ids if i in locks]
    if not pinned or not c.feasible:
        return c
    first = pinned[0]
    lock = locks[first]
    p = c.child_patches.get(first)
    if p is None or not _matches(p, lock):
        return GroupConfig(c.id + ":lock-infeasible", c.group_id, c.width, c.height, c.child_patches, c.source_order,
                           {**c.penalties, "infeasible": 1.0, "reason_lock_size": 1.0})
    ox, oy = float(lock.get("x", p["x"])) - p["x"], float(lock.get("y", p["y"])) - p["y"]
    for other in pinned[1:]:
        q = c.child_patches.get(other)
        lk = locks[other]
        if q is None or not _matches(q, lk) or abs(q["x"] + ox - float(lk.get("x", q["x"] + ox))) >= LOCK_TOLERANCE \
                or abs(q["y"] + oy - float(lk.get("y", q["y"] + oy))) >= LOCK_TOLERANCE:
            return GroupConfig(c.id + ":lock-infeasible", c.group_id, c.width, c.height, c.child_patches, c.source_order,
                               {**c.penalties, "infeasible": 1.0, "reason_lock_conflict": 1.0})
    return GroupConfig(c.id + ":anchored", c.group_id, c.width, c.height, c.child_patches, c.source_order,
                       {**c.penalties, "anchored": 1.0, "origin_x": ox, "origin_y": oy})


def _lock_derived(policy, members, locks, target, sw, sh) -> list[GroupConfig]:
    """For rigid units, the scale implied by a locked child's size, when it is
    uniform (aspect within 1%); otherwise an explicit infeasible record."""
    if policy.mode != "rigid":
        return []
    pinned = [i for i in policy.member_ids if i in locks]
    if not pinned:
        return []
    m = next(e for e in members if e["id"] == pinned[0])
    lock = locks[pinned[0]]
    b = element_box(m)
    kx = float(lock.get("width", b.w)) / max(1e-6, b.w)
    ky = float(lock.get("height", b.h)) / max(1e-6, b.h)
    if abs(math.log(max(1e-9, kx) / max(1e-9, ky))) > LOCK_TOLERANCE:
        return [GroupConfig(f"{policy.group_id}:lock-nonuniform", policy.group_id, 0.0, 0.0, {}, policy.source_order,
                            {"infeasible": 1.0, "reason_lock_nonuniform": 1.0})]
    s = kx
    patches, U, read = _rigid_patches(members, s)
    # Exact pin: overwrite the locked child's patch with its lock geometry.
    pl = patches[m["id"]]
    pl.update({k: float(lock[k]) for k in ("width", "height") if k in lock})
    c = GroupConfig(f"{policy.group_id}:lock@{s:.3f}", policy.group_id, U.w * s, U.h * s, patches, policy.source_order,
                    {"readability": round(read, 4), "scale": s})
    return [_apply_locks(c, policy, members, locks)]
