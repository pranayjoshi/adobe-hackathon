"""Fit-aware layout retrieval (plan 05, CHAMELEON_RETRIEVAL_VERSION=fit_aware).

1. Structural pre-filter: training-normalized features, explicit
   incompatibility filters (aspect, slot capacity per role class), pool of 32.
2. Block-to-slot assignment (Hungarian over capacity-expanded slots): role
   compatibility, hierarchy rank, log aspect distortion, measured text
   capacity, photo-crop constraint; impossible semantics are +inf (never
   selectable) and every source unit must be covered.
3. Measured materialization: each slot is laid out with the same
   renderer-accurate column solver the templates use; image-hosted copy stays
   inside its host; cover-allowed backgrounds fill the canvas.
4. Rerank by actual fit (unreadable text, new collisions, broken support,
   headline readability) and keep a few geometrically distinct arrangements.
Every rejected library arrangement records its reason.
"""
from __future__ import annotations

import copy
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .library import LibraryError, FEATURE_NAMES, load_library

POOL = 32
MATERIALIZE = 8
RETURN = 2  # each retrieved design is its own pipeline family; do not crowd refine seeds
MAX_LOG_RATIO = 0.6
INF = math.inf


@dataclass
class Unit:
    id: str
    cls: str                      # text | action | visual
    role: str
    blocks: list                  # blocks laid out in this unit (reading order)
    order: int
    box: object                   # source union Box
    host: object = None           # compound: image block hosting `blocks`
    rel_x: float | None = None    # host-relative x for copy over a detailed photo background
    font: float = 0.0
    ids: list = field(default_factory=list)


def _env_path():
    return Path(os.environ.get("CHAMELEON_LAYOUT_LIBRARY_V2", str(Path(__file__).resolve().parents[2] / "engine/models/layouts.v2.json")))


def _hosts(source, A=None):
    """Shared alpha-aware photo relationships; flat overlays are not hosts."""
    from ..photo_analysis import source_photo_supports
    return source_photo_supports(source, A)


def _reading_order(source, A):
    """C1 graph reads_before order when available; otherwise analysis order."""
    order = {b.id: i for i, b in enumerate(A.blocks)}
    graph = getattr(A, "graph", None)
    edges = getattr(graph, "edges", None) if graph is not None else None
    if edges:
        owner = {m["id"]: b.id for b in A.blocks for m in b.members}
        before = Counter()
        for e in edges:
            if isinstance(e, dict) and e.get("kind") == "reads_before":
                a, b = owner.get(e.get("a")), owner.get(e.get("b"))
                if a and b and a != b:
                    before[b] += 1
        if before:
            order = {bid: i for i, bid in enumerate(sorted(order, key=lambda k: (before[k], order[k])))}
    return order


def build_units(source, A):
    from ..canonical import union_all
    from ..scene import background_cover_allowed, low_detail_asset
    hosts = _hosts(source, A)
    order = _reading_order(source, A)
    by_el = {e["id"]: e for e in source["elements"]}
    block_of = {m["id"]: b for b in A.blocks for m in b.members}
    bg = {e["id"] for e in A.background}
    sh = A.sh
    compound: dict[str, list] = {}
    for b in A.blocks:
        if b.role in ("hero", "image", "logo"):
            continue
        texts = [m for m in b.members if m["type"] == "text"]
        if not texts or len(texts) != len(b.members):
            continue
        hs = {hosts.get(m["id"]) for m in texts}
        if len(hs) == 1:
            h = hs.pop()
            hb = block_of.get(h)
            if h and h not in bg and hb is not None and type(hb).__name__ == "ImageBlock" and hb is not b:
                compound.setdefault(hb.id, []).append(b)
    # Copy printed on a panel (analysis.panels) travels with that panel: the
    # panel becomes the compound's host, its text is laid out inside it.
    from ..blocks import RigidBlock
    claimed = {x.id for v in compound.values() for x in v}
    panel_units = []
    for pnl in A.panels:
        el = by_el.get(pnl["id"])
        kids = set(pnl.get("children") or ())
        members = [b for b in A.blocks if b.id not in claimed and b.role not in ("hero", "image", "logo")
                   and all(m["type"] == "text" for m in b.members) and set(b.ids) <= kids]
        if el is None or not members or el.get("rotation"):
            continue
        claimed.update(b.id for b in members)
        host = RigidBlock([el], el["id"], "image", "Panel", 0.4)
        members.sort(key=lambda x: order[x.id])
        fonts = [float(m["font_size"]) for x in members for m in x.members]
        first = min(order[x.id] for x in members)
        panel_units.append(Unit(host.id, "visual", "image", members, first, union_all([host.src_box] + [x.src_box for x in members]),
                                host=host, font=max(fonts, default=0) / sh, ids=[el["id"]] + [i for x in members for i in x.ids]))
    units, used = list(panel_units), set(claimed) - {x.id for v in compound.values() for x in v}
    for b in A.blocks:
        if b.id in used:
            continue
        if b.id in compound:
            members = sorted(compound[b.id], key=lambda x: order[x.id])
            used.update(x.id for x in members)
            box = union_all([b.src_box] + [x.src_box for x in members])
            fonts = [float(m["font_size"]) for x in members for m in x.members if m["type"] == "text"]
            units.append(Unit(b.id, "visual", b.role, members, order[b.id], box, host=b,
                              font=max(fonts, default=0) / sh, ids=list(b.ids) + [i for x in members for i in x.ids]))
            continue
        kind = type(b).__name__
        if kind == "ImageBlock":
            cls = "visual"
        elif kind in ("ButtonBlock", "RowBlock") or b.role in ("cta", "badge", "badges"):
            cls = "action"
        elif kind == "RigidBlock":
            cls = "visual" if any(m["type"] == "image" for m in b.members) else "text"
        else:
            cls = "text"
        fonts = [float(m["font_size"]) for m in b.members if m["type"] == "text"]
        rel = None
        texts = [m for m in b.members if m["type"] == "text"]
        h = hosts.get(texts[0]["id"]) if texts else None
        if h in bg and not low_detail_asset(by_el[h].get("asset_id")) and background_cover_allowed(source, by_el[h], {**by_el[h], "fit": "cover"}):
            from ..canonical import element_box
            hb = element_box(by_el[h])
            rel = (b.src_box.cx - hb.x) / max(1.0, hb.w)
        units.append(Unit(b.id, cls, b.role, [b], order[b.id], b.src_box, rel_x=rel,
                          font=max(fonts, default=0) / sh, ids=list(b.ids)))
    units.sort(key=lambda u: u.order)
    return units


def layered_composition(A):
    """(image block, panel) pairs where a content image sits mostly on a text panel."""
    from ..canonical import Box
    out = []
    for pnl in A.panels:
        if not pnl.get("children"):
            continue
        pb = Box(pnl["box"]["x"], pnl["box"]["y"], pnl["box"]["w"], pnl["box"]["h"])
        for b in A.blocks:
            if b.role in ("hero", "image") and b.src_box.intersection_area(pb) >= .5 * max(1e-6, b.src_box.area):
                out.append([b.id, pnl["id"]])
    return out


def query_features(source, A, units, target):
    texts = [u for u in units if u.cls == "text"]
    fonts = [u.font for u in units if u.font > 0]
    area = sum(u.box.w * u.box.h for u in texts) / (A.sw * A.sh)
    chars = sum(len(str(m.get("content", ""))) for u in texts for b in u.blocks for m in b.members if m["type"] == "text")
    return (math.log(target["width"] / target["height"]), len(texts), sum(u.cls == "action" for u in units),
            sum(u.cls == "visual" for u in units), area, max(fonts, default=0.0),
            (max(fonts) / max(1e-6, min(fonts))) if fonts else 1.0, 1, math.log1p(chars))


WEIGHTS = (4.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.3, 0.2, 0.5)


def _capacity(slot, n_text_units, n_text_slots):
    if slot["cls"] == "visual":
        return 1
    if slot["cls"] == "action":
        return 2
    return min(6, max(2, math.ceil(n_text_units / max(1, n_text_slots)) + 1))


def _allowed(u, s):
    """Role-compatibility cost, or None when the assignment is semantically impossible."""
    if u.cls == "text":
        return (0.0 if s["role"] == u.role else 0.3) if s["cls"] == "text" else None
    if u.cls == "action":
        return {"action": 0.0, "text": 0.9}.get(s["cls"])
    if u.role == "logo" and u.host is None:
        return {"visual": 0.0 if s["role"] == "logo" else 0.3, "text": 1.5}.get(s["cls"])
    return (0.0 if s["role"] == u.role else 0.3) if s["cls"] == "visual" else None


def assign(units, entry, target, A):
    """Hungarian assignment over capacity-expanded slots.

    Returns (assignment {unit_index: slot_index}, cost) or raises ValueError
    with a rejection reason. Forbidden pairs are +inf; scipy rejects an
    infeasible matrix rather than picking an impossible pair.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    slots = entry["slots"]
    cw, ch = float(target["width"]), float(target["height"])
    n_text_units = sum(u.cls in ("text", "action") for u in units)
    n_text_slots = sum(s["cls"] in ("text", "action") for s in slots)
    cols = [(j, c) for j, s in enumerate(slots) for c in range(_capacity(s, n_text_units, n_text_slots))]
    # Explicit coverage checks give readable rejection reasons before solving.
    if not slots:
        raise ValueError("insufficient_slots")
    for u in units:
        if not any(_allowed(u, s) is not None for s in slots):
            raise ValueError("impossible_semantic_assignment")
    vis_units = sum(1 for u in units if u.cls == "visual" and not (u.role == "logo" and u.host is None))
    if vis_units > sum(s["cls"] == "visual" for s in slots):
        raise ValueError("insufficient_slots")
    if len(units) > len(cols):
        raise ValueError("insufficient_slots")
    text_rank = {id(u): i for i, u in enumerate(sorted([u for u in units if u.font > 0], key=lambda u: -u.font))}
    s_fonts = sorted({s.get("font", 0) for s in slots if s.get("font")}, reverse=True)
    uniform = min(cw / A.sw, ch / A.sh)
    M = np.full((len(units), len(cols)), INF)
    for i, u in enumerate(units):
        ucx, ucy = u.box.cx / A.sw, u.box.cy / A.sh
        for k, (j, c) in enumerate(cols):
            s = slots[j]
            base = _allowed(u, s)
            if base is None:
                continue
            cost = base + 0.35 * c
            cost += 0.4 * (abs(ucx - (s["x"] + s["w"] / 2)) + abs(ucy - (s["y"] + s["h"] / 2)))
            sw_px, sh_px = max(1.0, s["w"] * cw), max(1.0, s["h"] * ch)
            if u.cls == "visual":
                cost += 0.5 * abs(math.log(max(1e-3, u.box.w / max(1e-3, u.box.h)) / (sw_px / sh_px)))
            else:
                need = max(1.0, u.box.w * u.box.h * uniform * uniform)
                cost += 0.25 * min(3.0, abs(math.log(need / (sw_px * sh_px))))
                if id(u) in text_rank and s.get("font") and s_fonts:
                    ru = text_rank[id(u)] / max(1, len(text_rank) - 1)
                    rs = s_fonts.index(s["font"]) / max(1, len(s_fonts) - 1)
                    cost += 0.6 * abs(ru - rs)
            if u.rel_x is not None:  # copy over a detailed photo keeps its crop position
                cost += 3.0 * max(0.0, abs(u.rel_x - (s["x"] + s["w"] / 2)) - 0.10)
            M[i, k] = cost
    try:
        rows, cs = linear_sum_assignment(M)
    except ValueError as exc:
        raise ValueError("impossible_semantic_assignment") from exc
    if len(rows) != len(units) or not np.all(np.isfinite(M[rows, cs])):
        raise ValueError("impossible_semantic_assignment")
    result = {int(i): cols[int(k)][0] for i, k in zip(rows, cs)}
    total = float(M[rows, cs].sum())
    # Group topology: reading order should not run backwards across slots.
    slot_rank = {j: n for n, j in enumerate(range(len(slots)))}
    inv = 0
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            if units[a].cls == units[b].cls == "text" and slot_rank[result[a]] > slot_rank[result[b]]:
                inv += 1
    return result, total + 0.3 * inv


def _expand(rects, cw, ch, m):
    """Grow each occupied slot toward free space (bounded), never into another slot."""
    from ..canonical import Box
    out = []
    for i, r in enumerate(rects):
        others = [o for j, o in enumerate(rects) if j != i]
        x0, y0, x1, y1 = r.x, r.y, r.x1, r.y1
        gx, gy = max(0.5 * r.w, 0.12 * cw), max(0.5 * r.h, 0.12 * ch)
        lim_l, lim_r, lim_t, lim_b = max(m, x0 - gx), min(cw - m, x1 + gx), max(m, y0 - gy), min(ch - m, y1 + gy)
        for o in others:
            vert = o.y < y1 and o.y1 > y0
            horiz = o.x < x1 and o.x1 > x0
            if vert and o.x1 <= x0 + 1e-6:
                lim_l = max(lim_l, o.x1 + m * 0.5)
            if vert and o.x >= x1 - 1e-6:
                lim_r = min(lim_r, o.x - m * 0.5)
            if horiz and o.y1 <= y0 + 1e-6:
                lim_t = max(lim_t, o.y1 + m * 0.5)
            if horiz and o.y >= y1 - 1e-6:
                lim_b = min(lim_b, o.y - m * 0.5)
        x0, x1 = min(x0, max(m, lim_l)), max(x1, min(cw - m, lim_r))
        y0, y1 = min(y0, max(m, lim_t)), max(y1, min(ch - m, lim_b))
        x0, y0 = max(0.0, x0), max(0.0, y0)
        out.append(Box(x0, y0, max(1.0, min(cw, x1) - x0), max(1.0, min(ch, y1) - y0)))
    return out


def _subtract(r, o, m):
    """Largest of the four sub-rectangles of r that avoid o (r itself if disjoint)."""
    from ..canonical import Box
    if r.intersection_area(o) <= 0:
        return r
    opts = [Box(r.x, r.y, max(0.0, o.x - m * 0.5 - r.x), r.h), Box(o.x1 + m * 0.5, r.y, max(0.0, r.x1 - o.x1 - m * 0.5), r.h),
            Box(r.x, r.y, r.w, max(0.0, o.y - m * 0.5 - r.y)), Box(r.x, o.y1 + m * 0.5, r.w, max(0.0, r.y1 - o.y1 - m * 0.5))]
    best = max(opts, key=lambda b: b.area)
    return best if best.area >= 0.08 * r.area and best.w >= 4 and best.h >= 4 else r


def materialize(source, target, A, units, entry, result):
    from ..canonical import Box
    from ..layout import Env, Placement, fit_column, fit_single, place_backdrop, place_column, place_corners
    from ..pipeline import serialise
    from ..scene import background_cover_allowed
    cw, ch = float(target["width"]), float(target["height"])
    env = Env(A, cw, ch)
    m = max(4.0, 0.03 * min(cw, ch))
    P = Placement("retrieved", "Retrieved arrangement", "Matched to training geometry")
    place_backdrop(env, P)
    cover = []
    for e in A.background:
        if e["type"] == "image" and background_cover_allowed(source, e, {**e, "fit": "cover"}):
            P.patches[e["id"]] = {"x": 0.0, "y": 0.0, "width": cw, "height": ch}
            cover.append(e["id"])
    slots = entry["slots"]
    occupied = sorted(set(result.values()))
    raw = []
    for j in occupied:
        s = slots[j]
        x, y = max(m, s["x"] * cw), max(m, s["y"] * ch)
        x1, y1 = min(cw - m, (s["x"] + s["w"]) * cw), min(ch - m, (s["y"] + s["h"]) * ch)
        raw.append(Box(min(x, cw - m - 1), min(y, ch - m - 1), max(1.0, x1 - x), max(1.0, y1 - y)))
    rects = dict(zip(occupied, _expand(raw, cw, ch, m)))
    # Library images often sit under copy; a visual slot yields to occupied copy slots.
    copy_slots = [j for j in occupied if any(units[i].cls != "visual" for i, jj in result.items() if jj == j)]
    for j in occupied:
        if j in copy_slots:
            continue
        for k in copy_slots:
            rects[j] = _subtract(rects[j], rects[k], m)
    area_k = math.sqrt(cw * ch / (A.sw * A.sh))
    for j in occupied:
        rect = rects[j]
        members = sorted([units[i] for i, jj in result.items() if jj == j], key=lambda u: u.order)
        if len(members) == 1 and members[0].cls == "visual":
            u = members[0]
            lead = u.host if u.host is not None else u.blocks[0]
            # Bounded growth: a logo/secondary image may not balloon into free space.
            s_cap = area_k * (2.0 if lead.role == "hero" or u.host is not None else 1.3)
            laid, _ = fit_single(env, lead, rect.w, rect.h, max(1e-3, min(max(env.s_hi, 2.0), s_cap)))
            hx, hy = rect.x + (rect.w - laid.w) / 2, rect.y + (rect.h - laid.h) / 2
            patches, boxes = laid.place(hx, hy)
            P.add(patches, boxes, lead.id)
            if u.host is not None:
                inset = 0.06 * min(laid.w, laid.h)
                inner = Box(hx + inset, hy + inset, max(1.0, laid.w - 2 * inset), max(1.0, laid.h - 2 * inset))
                fit = fit_column(env, u.blocks, inner)
                place_column(P, fit, u.blocks, inner, "center")
            continue
        blocks = [b for u in members for b in (([u.host] if u.host is not None else []) + u.blocks)]
        fit = fit_column(env, blocks, rect)
        place_column(P, fit, blocks, rect, "center")
    place_corners(env, P, A.stickers + A.accents)
    from .templates import min_visible
    min_visible(P)
    design = serialise(source, target, P)
    by = {e["id"]: e for e in design["elements"]}
    for eid in cover:
        by[eid].update(x=0.0, y=0.0, width=float(target["width"]), height=float(target["height"]), fit="cover")
    from ..repair import contain
    return contain(design, A)


def measure_fit(source, design, A, units):
    """Cheap measured fit: judge-floor readability, new collisions, broken support."""
    from ..blocks import FLOORS
    from ..canonical import element_box, rotated_extent
    from ..quality import collisions
    cw, ch = design["canvas"]["width"], design["canvas"]["height"]
    area = math.sqrt(cw * ch / (A.sw * A.sh))
    src = {e["id"]: e for e in source["elements"]}
    by = {e["id"]: e for e in design["elements"]}
    unread, headline = [], None
    for e in design["elements"]:
        if e["type"] != "text" or not str(e.get("content", "")).strip():
            continue
        role = A.roles.get(e["id"], "body")
        floor = min(FLOORS.get(role, 9), float(src[e["id"]]["font_size"]) * max(.5, min(1, area)))
        r = e["font_size"] / max(1e-6, floor)
        if r < .98:
            unread.append(e["id"])
        if role == "headline":
            headline = r if headline is None else min(headline, r)
    broken = 0
    for u in units:
        if u.host is None:
            continue
        hb = element_box(by[u.host.ids[0]])
        for b in u.blocks:
            for i in b.ids:
                bx = rotated_extent(by[i])
                if hb.intersection_area(bx) < .94 * max(1e-6, bx.area):
                    broken += 1
    displaced = sum(1 for u in units if u.rel_x is not None
                    and abs(u.rel_x - (sum(rotated_extent(by[i]).cx for i in u.ids) / len(u.ids)) / cw) > .14)
    coll = len(collisions(source, design, A))
    from ..quality import diagnose
    contrast = len(diagnose(source, design, A)["contrast_regressions"])
    return {"unreadable": len(unread), "collisions": coll, "contrast": contrast, "broken_support": broken, "displaced": displaced,
            "headline_ratio": round(headline if headline is not None else 1.0, 3)}


def _defects(m):
    return m["unreadable"] + m["collisions"] + m["broken_support"] + m["displaced"] + m["contrast"]


def retrieve_fit_aware(source, target, A, *, library_path=None):
    from .diversity import fingerprint, mmr_select
    t0 = time.monotonic()
    path = Path(library_path) if library_path else _env_path()
    if not path.is_file():
        raise LibraryError("missing_library", str(path))
    lib = load_library(path)
    units = build_units(source, A)
    report = {"version": "fit_aware", "library": lib.get("entries_sha256", "")[:16], "units": len(units),
              "rejections": Counter(), "rejected": [], "selected": []}
    layered = layered_composition(A)
    if layered:
        # An image printed on a panel is one layered picture; slot retrieval
        # would pull it apart (the incompatible-group rejection of plan 05).
        report["rejections"] = {"layered_composition": len(lib["entries"])}
        report["layered"] = layered
        report["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return [], report
    if not units:
        report["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return [], report
    q = query_features(source, A, units, target)
    mean, std = lib["normalization"]["mean"], lib["normalization"]["std"]
    zq = [(v - mu) / sd for v, mu, sd in zip(q, mean, std)]
    rt = target["width"] / target["height"]
    scored = []
    for e in lib["entries"]:
        if abs(math.log(rt / e["ratio"])) > MAX_LOG_RATIO:
            report["rejections"]["incompatible_ratio"] += 1
            continue
        ze = [(v - mu) / sd for v, mu, sd in zip(e["features"], mean, std)]
        dist = sum(w * abs(a - b) for w, a, b in zip(WEIGHTS, zq, ze))
        scored.append((dist, e["layout_id"], e))
    scored.sort(key=lambda t: (t[0], t[1]))
    pool = scored[:POOL]
    assigned = []
    for dist, lid, e in pool:
        try:
            result, cost = assign(units, e, target, A)
        except ValueError as exc:
            reason = str(exc) if str(exc) in ("insufficient_slots", "impossible_semantic_assignment") else "assignment_failed"
            report["rejections"][reason] += 1
            report["rejected"].append({"layout_id": lid, "reason": reason})
            continue
        assigned.append((cost / len(units) + 0.3 * dist, lid, e, result))
    assigned.sort(key=lambda t: (t[0], t[1]))
    built = []
    for score, lid, e, result in assigned[:MATERIALIZE]:
        try:
            design = materialize(source, target, A, units, e, result)
        except (ValueError, ZeroDivisionError, KeyError) as exc:
            report["rejections"]["materialize_failed"] += 1
            report["rejected"].append({"layout_id": lid, "reason": "materialize_failed", "detail": repr(exc)[:120]})
            continue
        m = measure_fit(source, design, A, units)
        built.append({"design": design, "layout_id": lid, "family": e["source_family"], "ratio": e["ratio"],
                      "assignment": round(score, 3), **m})
    for b in assigned[MATERIALIZE:]:
        report["rejections"]["not_materialized_budget"] += 1
    if built:
        best_head = max(b["headline_ratio"] for b in built)
        kept = []
        for b in built:
            if b["headline_ratio"] < .98 <= best_head:
                report["rejections"]["tiny_headline"] += 1
                report["rejected"].append({"layout_id": b["layout_id"], "reason": "tiny_headline", "headline_ratio": b["headline_ratio"]})
                continue
            kept.append(b)
        built = kept
    picked = mmr_select(built, RETURN, quality=lambda b: -b["assignment"] + 0.5 * min(2.0, b["headline_ratio"]),
                        fingerprint_of=lambda b: fingerprint(b["design"], A),
                        gate=lambda b: (-_defects(b), -b["unreadable"]))
    for b in built:
        if b not in picked:
            report["rejections"]["outranked"] += 1
    report["selected"] = [{k: v for k, v in b.items() if k != "design"} for b in picked]
    report["rejections"] = dict(report["rejections"])
    report["pool"] = len(pool)
    report["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
    return [b["design"] for b in picked], report
