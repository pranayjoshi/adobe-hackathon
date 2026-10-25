"""Structural fingerprints and maximal-marginal-relevance shortlist selection.

Template labels are metadata, not visual diversity: two templates can emit the
same geometry and one template can emit very different arrangements. The
fingerprint is computed from geometry only -- normalized centers/sizes of each
top-level layout unit, pairwise reading-order relations and (when a provider
supplies them) group configuration IDs. Selection keeps hard gates dominant:
an item from a worse gate tier is never chosen while a better tier still has
an unselected, non-duplicate item.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

FINGERPRINT_VERSION = "structural-fingerprint/1"
# Calibrated on the legacy portfolio: bias variants that saturate at the same
# scale differ by <0.004; visibly different arrangements are >0.05 apart.
NEAR_DUPLICATE = 0.012


@dataclass(frozen=True)
class Fingerprint:
    version: str
    units: tuple[tuple[str, float, float, float, float, float], ...]  # (unit id, cx, cy, w, h, weight)
    relations: tuple[tuple[str, str, str], ...]                        # (a, b, relation)
    configuration_ids: frozenset = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "units": [list(u) for u in self.units],
                "relations": [list(r) for r in self.relations], "configuration_ids": sorted(self.configuration_ids)}


def _relation(a, b, tol):
    """Coarse spatial relation between two normalized boxes (cx, cy, w, h)."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    dx, dy = bx - ax, by - ay
    if abs(dx) * 2 < (aw + bw) * 0.5 and abs(dy) * 2 < (ah + bh) * 0.5:
        return "overlap"
    if abs(dy) >= abs(dx):
        return "above" if dy > tol else ("below" if dy < -tol else "level")
    return "left_of" if dx > tol else ("right_of" if dx < -tol else "level")


def fingerprint(design: dict, analysis=None, configuration_ids: Iterable[str] = ()) -> Fingerprint:
    """Geometry-only fingerprint; uses layout units when an Analysis is given."""
    from ..canonical import rotated_extent, union_all
    cw, ch = float(design["canvas"]["width"]), float(design["canvas"]["height"])
    by = {e["id"]: e for e in design["elements"]}
    groups: list[tuple[str, list[str], float]] = []
    if analysis is not None:
        for b in analysis.all_blocks():
            groups.append((b.id, b.ids, float(getattr(b, "importance", 0.5))))
        ambient = [e["id"] for e in list(analysis.background) + list(analysis.decorations)]
        groups += [(i, [i], 0.15) for i in ambient]
    else:
        for e in design["elements"]:
            groups.append((e["id"], [e["id"]], 1.0 if e.get("type") == "text" else 0.6))
    units = []
    for gid, ids, weight in groups:
        boxes = [rotated_extent(by[i]) for i in ids if i in by]
        if not boxes:
            continue
        u = union_all(boxes)
        units.append((gid, round(u.cx / cw, 4), round(u.cy / ch, 4), round(u.w / cw, 4), round(u.h / ch, 4), weight))
    units.sort(key=lambda t: t[0])
    main = [u for u in units if u[5] > 0.15]
    rel = []
    for i, a in enumerate(main):
        for b in main[i + 1:]:
            rel.append((a[0], b[0], _relation(a[1:5], b[1:5], 0.02)))
    return Fingerprint(FINGERPRINT_VERSION, tuple(units), tuple(rel), frozenset(configuration_ids))


def distance(a: Fingerprint, b: Fingerprint) -> float:
    """0 = identical geometry. Weighted unit displacement + relation/config changes."""
    ua, ub = {u[0]: u for u in a.units}, {u[0]: u for u in b.units}
    shared = set(ua) & set(ub)
    total_w = sum(ua[k][5] for k in shared) or 1.0
    geo = 0.0
    for k in shared:
        x, y = ua[k], ub[k]
        geo += x[5] * (abs(x[1] - y[1]) + abs(x[2] - y[2]) + 0.5 * (abs(x[3] - y[3]) + abs(x[4] - y[4])))
    geo /= total_w
    ra, rb = {r[:2]: r[2] for r in a.relations}, {r[:2]: r[2] for r in b.relations}
    keys = set(ra) | set(rb)
    rel = sum(ra.get(k) != rb.get(k) for k in keys) / len(keys) if keys else 0.0
    ca, cb = a.configuration_ids, b.configuration_ids
    cfg = (1 - len(ca & cb) / len(ca | cb)) if (ca or cb) else 0.0
    unmatched = len(set(ua) ^ set(ub)) / max(1, len(set(ua) | set(ub)))
    return geo + 0.25 * rel + 0.1 * cfg + unmatched


def dedupe(items: Sequence, fingerprint_of: Callable[[Any], Fingerprint], threshold: float = NEAR_DUPLICATE) -> list:
    """Keep the first of each group of near-identical geometries (input order wins)."""
    kept, prints = [], []
    for item in items:
        fp = fingerprint_of(item)
        if any(distance(fp, q) < threshold for q in prints):
            continue
        kept.append(item)
        prints.append(fp)
    return kept


def mmr_select(items: Sequence, k: int, *, quality: Callable[[Any], float],
               fingerprint_of: Callable[[Any], Fingerprint], gate: Callable[[Any], Any] | None = None,
               lam: float = 0.7, scale: float = 0.15, near_duplicate: float = NEAR_DUPLICATE) -> list:
    """Maximal-marginal-relevance shortlist under dominant hard gates.

    `gate(item)` returns a comparable tier key (higher is better, e.g. the
    first elements of the pipeline rank tuple). Items are only drawn from the
    best tier that still has selectable items, so diversity can never promote
    an invalid or defect-heavier candidate over a better-gated one. Within a
    tier, score = lam * normalized quality - (1 - lam) * similarity to the
    closest already selected item; near-duplicates of a chosen item come after
    every distinct item of the same tier (and before any worse tier).
    """
    items = list(items)
    if k <= 0 or not items:
        return []
    gate = gate or (lambda _: 0)
    prints = [fingerprint_of(x) for x in items]
    quals = [float(quality(x)) for x in items]
    lo, hi = min(quals), max(quals)
    norm = [(q - lo) / (hi - lo) if hi > lo else 1.0 for q in quals]
    tiers = sorted({gate(x) for x in items}, reverse=True)
    chosen: list[int] = []
    for tier in tiers:
        pool = [i for i in range(len(items)) if gate(items[i]) == tier]
        duplicates: list[int] = []
        while pool and len(chosen) < k:
            best, best_score = None, -math.inf
            for i in pool:
                d = min((distance(prints[i], prints[j]) for j in chosen), default=math.inf)
                if d < near_duplicate:
                    continue
                sim = 0.0 if d == math.inf else max(0.0, 1.0 - d / scale)
                s = lam * norm[i] - (1 - lam) * sim
                if s > best_score + 1e-12:
                    best, best_score = i, s
            if best is None:  # only near-duplicates of chosen items left in this tier
                duplicates = pool
                break
            chosen.append(best)
            pool.remove(best)
        # Strict gate dominance: a better-tier near-duplicate still precedes
        # any worse-tier item, but only after every distinct item of its tier.
        for i in sorted(duplicates, key=lambda i: -quals[i]):
            if len(chosen) >= k:
                break
            chosen.append(i)
        if len(chosen) >= k:
            break
    return [items[i] for i in chosen]
