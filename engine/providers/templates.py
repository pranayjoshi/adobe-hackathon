"""Diverse template portfolio: geometry dedupe + measured, bounded variant search.

Called by engine.layout.propose when CHAMELEON_PROPOSAL_PORTFOLIO=diverse.
Variants are parameterized by measured content (renderer-accurate text
metrics through Block.layout) over a small grid -- column count, image side,
hero height share -- never a Cartesian product over every block. Only
variants on the fit / readability / fidelity Pareto frontier that are not
dominated by an existing legacy placement are added.
"""
from __future__ import annotations

import math
import time

from .diversity import NEAR_DUPLICATE, distance, fingerprint

EXACT_DUPLICATE = 5e-4  # template dedupe: geometrically identical only (bias variants that saturate)
MAX_ADDED = 2  # one per structural kind: pipeline refine seeds are scarce (3 families)


def _elements(A):
    els = {}
    for b in A.all_blocks():
        for m in b.members:
            els[m["id"]] = m
    for e in list(A.background) + list(A.decorations):
        els[e["id"]] = e
    return els


def placement_design(A, P, cw, ch) -> dict:
    """Geometry-only pseudo design for fingerprinting a Placement."""
    els = _elements(A)
    out = []
    for eid, patch in P.patches.items():
        src = els.get(eid, {})
        out.append({"id": eid, "type": src.get("type", "image"), "rotation": src.get("rotation", 0) or 0,
                    "x": patch["x"], "y": patch["y"], "width": patch["width"], "height": patch["height"],
                    **({"font_size": patch["font_size"]} if "font_size" in patch else {})})
    return {"canvas": {"width": cw, "height": ch}, "elements": out}


def placement_metrics(A, P, cw, ch) -> dict:
    """Measured readability (judge floor ratios), fill and fidelity of a placement."""
    from ..blocks import FLOORS
    from ..canonical import Box
    area = math.sqrt(cw * ch / (A.sw * A.sh))
    els = _elements(A)
    ratios, headline = [], None
    for eid, patch in P.patches.items():
        e = els.get(eid)
        if not e or e.get("type") != "text" or not str(e.get("content", "")).strip() or "font_size" not in patch:
            continue
        role = A.roles.get(eid, "body")
        floor = min(FLOORS.get(role, 9), float(e["font_size"]) * max(.5, min(1, area)))
        r = patch["font_size"] / max(1e-6, floor)
        ratios.append(r)
        if role == "headline":
            headline = r if headline is None else min(headline, r)
    canvas = Box(0, 0, cw, ch)
    boxes = [b for b in P.boxes.values()]
    fill = sum(b.intersection_area(canvas) for b in boxes) / max(1.0, canvas.area)
    disp = []
    for blk in A.blocks:
        b = P.boxes.get(blk.id)
        if b is None:
            continue
        sb = blk.src_box
        disp.append(abs(b.cx / cw - sb.cx / A.sw) + abs(b.cy / ch - sb.cy / A.sh))
    return {"unreadable": sum(r < 0.98 for r in ratios), "min_ratio": round(min(ratios, default=1.0), 3),
            "headline_ratio": round(headline if headline is not None else min(ratios, default=1.0), 3),
            "fill": round(min(1.0, fill), 3), "fidelity": round(1 - (sum(disp) / len(disp) if disp else 0) / 2, 3),
            "relaxed": bool(P.relaxed_floors)}


def _source_from(A):
    els = sorted(_elements(A).values(), key=lambda e: e.get("z_index", 0))
    return {"canvas": {"width": A.sw, "height": A.sh}, "elements": els}


def contrast_losses(A, P, cw, ch) -> int:
    """Estimated (non-rendered) contrast regressions of a placement, as the judge counts them."""
    from ..pipeline import serialise
    from ..quality import diagnose
    src = _source_from(A)
    design = serialise(src, {"width": int(cw), "height": int(ch)}, P)
    return len(diagnose(src, design, A)["contrast_regressions"])


def _objectives(m):
    return (-(m["unreadable"] + m.get("contrast", 0)), min(m["headline_ratio"], 3.0), min(m["min_ratio"], 1.5), m["fill"], m["fidelity"])


def _dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y + 1e-9 for x, y in zip(a, b))


def pareto(items, key):
    objs = [key(x) for x in items]
    return [x for i, x in enumerate(items) if not any(_dominates(objs[j], objs[i]) for j in range(len(items)) if j != i)]


def _variants(A, cw, ch, emphasis):
    from ..layout import Env, VISUAL_ROLES, tpl_flow, tpl_stack_hero
    env = Env(A, float(cw), float(ch), emphasis)
    rt = cw / ch
    out = []
    if rt >= 1.5:
        copy = [b for b in A.blocks if b.role not in VISUAL_ROLES]
        visuals = [b for b in A.blocks if b.role in VISUAL_ROLES]
        hero = A.block("hero") or (visuals[0] if visuals else None)
        src_side = "left" if hero is None or hero.src_box.cx <= A.sw / 2 else "right"
        sides = (src_side, "right" if src_side == "left" else "left") if visuals else (src_side,)
        max_cols = max(1, min(5, len(copy), int(cw // 110)))
        for cols in range(1, max_cols + 1):
            for side in sides:
                out.append(lambda c=cols, s=side: tpl_flow(env, c, s))
    if rt <= 0.8:
        for frac in (0.22, 0.32, 0.45):
            out.append(lambda f=frac: tpl_stack_hero(env, f))
    return out


def diverse_portfolio(A, cw, ch, emphasis, legacy, report) -> list:
    t0 = time.monotonic()
    prints, kept, dropped = [], [], []
    for P in legacy:
        fp = fingerprint(placement_design(A, P, cw, ch), A)
        # Near-identical (not identical) variants can refine differently, so keep them.
        if any(distance(fp, q) < EXACT_DUPLICATE for q in prints):
            dropped.append(P.template)
            continue
        prints.append(fp)
        kept.append(P)
    legacy_objs = [_objectives({**placement_metrics(A, P, cw, ch), "contrast": contrast_losses(A, P, cw, ch)}) for P in kept]
    candidates = []
    for make in _variants(A, cw, ch, emphasis):
        try:
            P = make()
        except (ValueError, ZeroDivisionError):
            P = None
        if P is None:
            continue
        m = placement_metrics(A, P, cw, ch)
        m["contrast"] = contrast_losses(A, P, cw, ch)
        candidates.append((P, m))
    front = pareto(candidates, key=lambda t: _objectives(t[1]))
    # Only variants that add something no legacy placement already offers.
    front = [t for t in front if not any(_dominates(o, _objectives(t[1])) for o in legacy_objs)]
    front.sort(key=lambda t: _objectives(t[1]), reverse=True)
    added, kinds = [], set()
    for P, m in front:
        if len(added) >= MAX_ADDED:
            break
        kind = P.template.split("_")[0]
        if kind in kinds:
            continue
        fp = fingerprint(placement_design(A, P, cw, ch), A)
        if any(distance(fp, q) < NEAR_DUPLICATE for q in prints):
            continue
        prints.append(fp)
        kinds.add(kind)
        kept.append(P)
        added.append({"template": P.template, **m})
    for P in kept:
        min_visible(P)
    report.update({"portfolio": "diverse", "legacy": len(legacy), "dropped_duplicates": dropped,
                   "searched": len(candidates), "added": added,
                   "elapsed_ms": round((time.monotonic() - t0) * 1000, 1)})
    return kept


def min_visible(P, floor: float = 1.05) -> None:
    """Thin source strokes must not scale below one pixel ("Nothing hidden").
    Grows the element uniformly about its centre; geometry only."""
    for patch in P.patches.values():
        w, h = patch.get("width", 0), patch.get("height", 0)
        small = min(w, h)
        if 0 < small < floor:
            k = floor / small
            patch.update(x=patch["x"] - w * (k - 1) / 2, y=patch["y"] - h * (k - 1) / 2, width=w * k, height=h * k)
