"""Transparent, explainable quality scoring for a candidate layout.

Four sub-scores (0-100) plus human-readable issues. Weights are engineering
defaults meant to be calibrated against blind human preferences later.
"""

from __future__ import annotations

import math
from typing import Any

from .analysis import Analysis
from .canonical import Box, element_box
from .layout import Placement
from .typography import measure

WEIGHTS = {"readability": 0.3, "hierarchy": 0.2, "composition": 0.3, "fidelity": 0.2}

# Comfortable sizes as (absolute px, fraction of the canvas short side).
IDEAL = {
    "headline": (18, 0.07), "subhead": (12, 0.035), "body": (11, 0.03), "cta": (12, 0.035),
    "badge": (10, 0.025), "badges": (10, 0.025), "sticker": (10, 0.025), "legal": (8, 0.018), "group": (10, 0.03),
}


def _issue(severity: str, message: str, ids: list[str] | None = None) -> dict[str, Any]:
    return {"severity": severity, "message": message, "ids": ids or []}


def score(source: dict[str, Any], output: dict[str, Any], A: Analysis, P: Placement) -> dict[str, Any]:
    cw, ch = float(output["canvas"]["width"]), float(output["canvas"]["height"])
    short = min(cw, ch)
    out = {e["id"]: e for e in output["elements"]}
    src = {e["id"]: e for e in source["elements"]}
    issues: list[dict[str, Any]] = []

    # --- Readability -------------------------------------------------------
    read_num = read_den = 0.0
    for eid, e in out.items():
        if e["type"] != "text":
            continue
        role = A.roles.get(eid, "body")
        role = "headline" if role in ("hero", "image", "logo") else role
        px, frac = IDEAL.get(role, IDEAL["body"])
        ideal = min(max(px, frac * short), float(src[eid]["font_size"]))
        fs = float(e["font_size"])
        w = {"headline": 3, "cta": 2, "subhead": 1.5}.get(role, 1.0)
        read_num += w * min(1.0, fs / ideal) ** 1.5
        read_den += w
        fit = measure(e, int(fs), float(e["width"]))
        if fit.overflow:
            issues.append(_issue("error", f"“{_short(e['content'])}” has a word wider than its box.", [eid]))
            read_num -= w * 0.5
        if fs < 7:
            issues.append(_issue("warn", f"{_role_label(role)} is {int(fs)}px — likely unreadable at this size.", [eid]))
        elif fs < ideal * 0.7:
            issues.append(_issue("info", f"{_role_label(role)} shrinks to {int(fs)}px (from {int(src[eid]['font_size'])}px).", [eid]))
    readability = 100 * max(0.0, read_num / read_den) if read_den else 100.0

    # --- Hierarchy ---------------------------------------------------------
    texts = [eid for eid in out if out[eid]["type"] == "text"]
    pairs = kept = 0
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            fa, fb = float(src[a]["font_size"]), float(src[b]["font_size"])
            if abs(fa - fb) / max(fa, fb) < 0.15:
                continue
            hi, lo = (a, b) if fa > fb else (b, a)
            pairs += 1
            kept += float(out[hi]["font_size"]) >= float(out[lo]["font_size"])
    order_score = kept / pairs if pairs else 1.0
    prom = 1.0
    hero = A.block("hero")
    if hero and hero.id in P.boxes:
        src_share = hero.src_box.area / (A.sw * A.sh)
        out_share = P.boxes[hero.id].area / (cw * ch)
        prom = min(1.0, out_share / max(1e-6, 0.45 * src_share))
        if prom < 0.5:
            issues.append(_issue("info", "Product image loses prominence at this size.", hero.ids))
    # Reward useful prominence relative to destination scale, with a bounded cap.
    ambition=[]
    for eid in texts:
        if A.roles.get(eid) in ('headline','cta'):
            desired=float(src[eid]['font_size'])*math.sqrt(cw*ch/(A.sw*A.sh))*1.1
            ambition.append(min(1,float(out[eid]['font_size'])/max(1,desired)))
    presence=sum(ambition)/len(ambition) if ambition else 1
    hierarchy = 100 * (0.55 * order_score + 0.25 * prom + .2*presence)
    if order_score < 1:
        issues.append(_issue("warn", "Type hierarchy is partly inverted (a secondary line outsizes a primary one)."))

    # --- Composition -------------------------------------------------------
    top = {bid: b for bid, b in P.boxes.items()}
    ids = list(top)
    src_boxes = {b.id: b.src_box for b in A.all_blocks()}
    overlap_pen = 0.0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            inter = top[a].intersection_area(top[b])
            if inter <= 0:
                continue
            smaller = min(top[a].area, top[b].area) or 1.0
            src_inter = src_boxes[a].intersection_area(src_boxes[b]) if a in src_boxes and b in src_boxes else 0
            if src_inter > 0:
                continue  # overlap existed by design
            frac = inter / smaller
            overlap_pen += frac
            if frac > 0.08:
                la, lb = _label(A, a), _label(A, b)
                issues.append(_issue("warn", f"{la} collides with {lb}.", []))
    fg = [b for b in top.values()]
    coverage = 0.0
    if fg:
        u = fg[0]
        for b in fg[1:]:
            u = u.union(b)
        coverage = min(1.0, u.clip(cw, ch).area / (cw * ch))
        cx = sum(b.cx * b.area for b in fg) / max(1e-6, sum(b.area for b in fg))
        cy = sum(b.cy * b.area for b in fg) / max(1e-6, sum(b.area for b in fg))
        balance = 1 - min(1.0, math.hypot((cx - cw / 2) / cw, (cy - ch / 2) / ch) * 2.5)
    else:
        balance = 1.0
    # Hull coverage can be large even when its interior is empty. Measure
    # occupied visible boxes independently; decorations/backgrounds excluded.
    from PIL import Image,ImageDraw
    from .quality import ink
    mask=Image.new('1',(160,160));draw=ImageDraw.Draw(mask)
    for e in output['elements']:
        if e['id'] not in {i for b in A.all_blocks() for i in b.ids}:continue
        b=ink(e)
        if b.w>0 and b.h>0 and b.x1>0 and b.y1>0 and b.x<cw and b.y<ch:draw.rectangle((max(0,b.x/cw*159),max(0,b.y/ch*159),min(159,b.x1/cw*159),min(159,b.y1/ch*159)),fill=1)
    density=mask.histogram()[1]/(160*160)
    space = .6*min(1.0, coverage / 0.6)+.4*min(1.0,density/.3)
    if coverage < 0.35:
        issues.append(_issue("info", "Content uses less than a third of the canvas — lots of empty space."))
    composition = 100 * max(0.0, 0.45 * space + 0.25 * balance + 0.3 * (1 - min(1.0, overlap_pen)))

    # --- Fidelity ----------------------------------------------------------
    # Relative order of blocks along each axis is preserved where the
    # aspect change allows it; penalties relax as the change grows.
    change = abs(math.log((cw / ch) / (A.sw / A.sh)))
    agree = total = 0
    blocks = [b for b in A.blocks if b.id in top]
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            sa, sb = a.src_box, b.src_box
            oa, ob = top[a.id], top[b.id]
            for s0, s1, o0, o1 in ((sa.cy, sb.cy, oa.cy, ob.cy), (sa.cx, sb.cx, oa.cx, ob.cx)):
                if abs(s0 - s1) < 0.05 * max(A.sw, A.sh):
                    continue
                total += 1
                agree += (s0 < s1) == (o0 < o1) or abs(o0 - o1) < 0.03 * max(cw, ch)
    order_keep = agree / total if total else 1.0
    tolerance = min(0.6, change * 0.4)
    fidelity = 100 * min(1.0, order_keep + tolerance)
    if P.relaxed_floors:
        issues.append(_issue("warn", "Not all copy fits at comfortable sizes — some text is set below its readability floor."))

    total_score = sum(WEIGHTS[k] * v for k, v in (("readability", readability), ("hierarchy", hierarchy),
                                                   ("composition", composition), ("fidelity", fidelity)))
    return {
        "total": round(total_score, 1),
        "parts": {
            "readability": round(readability, 1),
            "hierarchy": round(hierarchy, 1),
            "composition": round(composition, 1),
            "fidelity": round(fidelity, 1),
        },
        "issues": _dedupe(issues),
        "stats": {"coverage": round(coverage, 3), "ink_density":round(density,3), "min_font": min((float(out[t]["font_size"]) for t in texts), default=None)},
    }


def _short(s: str, n: int = 28) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _role_label(role: str) -> str:
    from .analysis import LABELS
    return LABELS.get(role, role.title())


def _label(A: Analysis, block_id: str) -> str:
    for b in A.all_blocks():
        if b.id == block_id:
            return b.label
    return block_id


def _dedupe(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for i in issues:
        if i["message"] in seen:
            continue
        seen.add(i["message"])
        out.append(i)
    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda i: order[i["severity"]])
