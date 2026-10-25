"""Preservation checks between a source design and a resized output.

These mirror the structural rules in RULES.md. A candidate that fails any
`error` check is never returned as the engine's answer when a passing one
exists.
"""

from __future__ import annotations

import math
from typing import Any

from .canonical import MUTABLE_FIELDS, element_box, rotated_extent

TOL = 0.5


def _check(name: str, ok: bool, detail: str = "", severity: str = "error") -> dict[str, Any]:
    return {"name": name, "passed": bool(ok), "detail": detail, "severity": severity}


def validate(source: dict[str, Any], output: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    required={"text":("content","font_family","font_size","color"),"image":("asset_id",),"shape":("shape_type","fill_color")}
    malformed=[str(e.get('id','?')) for e in output.get('elements',[]) if e.get('type') not in required or any(k not in e for k in required.get(e.get('type'),()))]
    checks.append(_check('Required fields present',not malformed,', '.join(malformed)))
    src = {e["id"]: e for e in source["elements"]}
    out_list = output.get("elements", [])
    out = {e.get("id"): e for e in out_list}
    cw, ch = output["canvas"]["width"], output["canvas"]["height"]

    missing = sorted(set(src) - set(out))
    extra = sorted(set(out) - set(src))
    dupes = len(out_list) != len(out)
    checks.append(_check(
        "Element IDs preserved",
        not missing and not extra and not dupes and len(out_list) == len(src),
        f"{len(src)}/{len(src)} elements" if not (missing or extra or dupes)
        else f"missing {missing}, unexpected {extra}" + (", duplicate ids" if dupes else ""),
    ))
    checks.append(_check('Element paint order preserved',
        [e.get('id') for e in out_list] == [e['id'] for e in source['elements']],
        'Original order retained, including equal-z layers'))

    from .scene import background_cover_allowed
    changed = []
    for eid, s in src.items():
        o = out.get(eid)
        if o is None:
            continue
        for k in set(s) | set(o):
            if k == "fit" and background_cover_allowed(source,s,o):continue
            if k in MUTABLE_FIELDS:
                continue
            if s.get(k) != o.get(k):
                changed.append(f"{eid}.{k}")
    content_changed = [c for c in changed if c.split(".")[1] in ("content", "font_family", "asset_id", "type")]
    checks.append(_check("Text, fonts & assets unchanged", not content_changed,
                         "identical" if not content_changed else ", ".join(content_changed[:6])))
    style_changed = [c for c in changed if c not in content_changed]
    checks.append(_check("Styles, layering & groups unchanged", not style_changed,
                         "identical" if not style_changed else ", ".join(style_changed[:6])))

    bad_nums = [
        eid for eid, o in out.items()
        if any(not isinstance(o.get(k), (int, float)) or isinstance(o.get(k), bool) or not math.isfinite(o.get(k)) for k in ("x", "y", "width", "height"))
        or o.get("width", 0) < 0 or o.get("height", 0) < 0
        or (o.get("type") == "text" and (not isinstance(o.get("font_size"), (int, float)) or not math.isfinite(o["font_size"]) or o["font_size"] < 1))
    ]
    checks.append(_check("Valid geometry", not bad_nums, "all finite" if not bad_nums else ", ".join(bad_nums)))

    off = []
    for eid, o in out.items():
        if eid in bad_nums:
            continue
        b = rotated_extent(o)
        if b.x < -TOL or b.y < -TOL or b.x1 > cw + TOL or b.y1 > ch + TOL:
            off.append(eid)
    checks.append(_check("Inside canvas", not off, f"{cw}×{ch}" if not off else ", ".join(off)))

    distorted = []
    for eid, s in src.items():
        o = out.get(eid)
        if not o or s.get("type") != "image" or s.get("fit") == "cover" or background_cover_allowed(source,s,o) or eid in bad_nums:
            continue
        if s["height"] <= 0 or o["height"] <= 0:
            continue
        r0, r1 = s["width"] / s["height"], o["width"] / o["height"]
        if abs(r1 / r0 - 1) > 0.05:
            distorted.append(eid)
    checks.append(_check("Image proportions kept", not distorted, "within 5%" if not distorted else ", ".join(distorted)))

    no_hide = [eid for eid, o in out.items() if eid not in bad_nums and (o["width"] < 1 or o["height"] < 1)]
    checks.append(_check("Nothing hidden", not no_hide, "all visible" if not no_hide else ", ".join(no_hide)))
    return checks


def passes(checks: list[dict[str, Any]]) -> bool:
    return all(c["passed"] for c in checks if c["severity"] == "error")
