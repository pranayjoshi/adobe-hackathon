"""Versioned, geometry-only layout library (plan 05, schema v2).

An entry is an arrangement, never content: normalized slot boxes with a role,
a role class, measured typography hints (normalized font size, character and
line counts -- not the characters themselves), grouping, provenance and the
family/split it came from. Held-out families (the visual development suite)
and near-duplicate clusters are excluded at build time; `validate_library`
re-checks every invariant at load so a contaminated, stale or incompatible
artifact fails closed to the template fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "chameleon-layout-library/2"
FEATURE_NAMES = ("log_ratio", "text_slots", "action_slots", "visual_slots", "text_area", "max_font",
                 "font_spread", "columns", "text_chars")
TEXT_ROLES = ("headline", "subhead", "body", "legal")
ACTION_ROLES = ("cta", "badge", "badges")
VISUAL_ROLES = ("hero", "image", "logo")
SUPPORTED_EMBEDDINGS = ()          # no visual encoder ships; any embedding block is incompatible
SPLIT_SALT = "chameleon-retrieval-split/1"
VALIDATION_PERCENT = 20
# Visual-suite development panel (plan 19): never in a library or its statistics.
VISUAL_SUITE_SOURCE_IDS = ("5dd7b76a9fea0cc374cc080a", "5ef30fa9499b85dcc73cb20e", "58891e5c95a7a863ddcc2f4b")
VISUAL_SUITE_CATALOG_IDS = ("b44ab672801fd912f6c2", "d958006f854dabe45ada", "f1cbff73226eeb998ad9")


class LibraryError(ValueError):
    """Invalid/incompatible library; callers fall back to templates with a diagnostic."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def role_class(role: str) -> str:
    if role in TEXT_ROLES:
        return "text"
    if role in ACTION_ROLES:
        return "action"
    if role in VISUAL_ROLES:
        return "visual"
    return "other"


def text_family(source: dict) -> str:
    """Exact-copy family key (same normalization as tools/bootstrap_roles.py)."""
    key = "|".join(sorted(re.sub(r"\s+", " ", str(e.get("content", ""))).strip().lower()
                          for e in source["elements"] if e.get("content")))
    return hashlib.sha256(key.encode()).hexdigest()[:20] if key else ""


def split_of(family: str) -> str:
    bucket = int(hashlib.sha256((SPLIT_SALT + family).encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def _clip01(v: float) -> float:
    return round(min(1.0, max(0.0, v)), 5)


def build_entry(source: dict, roles: dict | None = None, *, family: str, source_id: str = "",
                provenance: dict | None = None) -> dict | None:
    """Arrangement record for one design, or None when it has no usable copy slot."""
    roles = roles or {}
    W, H = float(source["canvas"]["width"]), float(source["canvas"]["height"])
    if W <= 0 or H <= 0:
        return None
    els = sorted(source["elements"], key=lambda e: e.get("z_index", 0))
    zmin = min((e.get("z_index", 0) for e in els), default=0)
    texts = [e for e in els if e.get("type") == "text" and str(e.get("content", "")).strip()]
    order = sorted(texts, key=lambda e: -float(e.get("font_size", 0)))
    rank = {e["id"]: i for i, e in enumerate(order)}
    top_fs = float(order[0]["font_size"]) if order else 0.0
    slots: list[dict[str, Any]] = []
    actions = []
    for e in els:
        x, y, w, h = float(e["x"]), float(e["y"]), float(e["width"]), float(e["height"])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 1 or h <= 1:
            continue
        cover = max(0.0, min(x + w, W) - max(x, 0)) * max(0.0, min(y + h, H) - max(y, 0)) / (W * H)
        weak = roles.get(e["id"])
        if e.get("type") != "text":
            if weak in ("background", "decoration") or (cover >= 0.85 and e.get("z_index", 0) <= zmin + 1):
                continue
            role = weak if weak in VISUAL_ROLES + ("cta", "badge") else None
            if role is None and 0.03 <= cover <= 0.7 and 1 / 8 <= w / h <= 8:
                role = "image"
            if role is None:
                continue
            slot = {"role": role, "cls": role_class(role), "x": x / W, "y": y / H, "w": w / W, "h": h / H,
                    "z": int(e.get("z_index", 0))}
            if slot["cls"] == "action":
                actions.append(slot)
            slots.append(slot)
            continue
        if e not in texts:
            continue
        cx, cy = x + w / 2, y + h / 2
        host = next((a for a in actions if a["x"] * W <= cx <= (a["x"] + a["w"]) * W and a["y"] * H <= cy <= (a["y"] + a["h"]) * H), None)
        content = str(e.get("content", ""))
        if host is not None:  # label of a button/badge shape: one action slot
            host["chars"] = host.get("chars", 0) + len(content)
            host["font"] = max(host.get("font", 0.0), float(e["font_size"]) / H)
            continue
        if weak in TEXT_ROLES + ACTION_ROLES:
            role = weak
        else:
            fs = float(e.get("font_size", 0))
            role = ("headline" if rank[e["id"]] == 0 else
                    "legal" if fs <= min(16.0, max(10.0, 0.35 * top_fs)) and cy > 0.75 * H else
                    "subhead" if rank[e["id"]] == 1 else "body")
        slots.append({"role": role, "cls": role_class(role), "x": x / W, "y": y / H, "w": w / W, "h": h / H,
                      "z": int(e.get("z_index", 0)), "font": float(e["font_size"]) / H, "chars": len(content),
                      "lines": content.count("\n") + 1})
    for s in slots:  # clip into the unit canvas
        x0, y0 = _clip01(s["x"]), _clip01(s["y"])
        x1, y1 = _clip01(s["x"] + s["w"]), _clip01(s["y"] + s["h"])
        s.update(x=x0, y=y0, w=round(max(0.0, x1 - x0), 5), h=round(max(0.0, y1 - y0), 5))
        if "font" in s:
            s["font"] = round(s["font"], 5)
    slots = [s for s in slots if s["w"] > 0.005 and s["h"] > 0.005]
    if not any(s["cls"] == "text" for s in slots) or len(slots) > 24:
        return None
    slots.sort(key=lambda s: (round(s["y"] / 0.06), s["x"]))
    groups = _groups(slots)
    entry = {"ratio": round(W / H, 5), "slots": slots, "groups": groups,
             "role_counts": {c: sum(s["cls"] == c for s in slots) for c in ("text", "action", "visual")},
             "source_family": family, "source_id": source_id, "split": split_of(family),
             "provenance": provenance or {}}
    entry["layout_id"] = hashlib.sha256(json.dumps({"r": entry["ratio"], "s": slots}, sort_keys=True).encode()).hexdigest()[:16]
    entry["features"] = list(raw_features(entry))
    return entry


def _groups(slots):
    """Column clusters of text slots (horizontal overlap, small vertical gap)."""
    idx = [i for i, s in enumerate(slots) if s["cls"] == "text"]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a in idx:
        for b in idx:
            if a >= b:
                continue
            sa, sb = slots[a], slots[b]
            ov = min(sa["x"] + sa["w"], sb["x"] + sb["w"]) - max(sa["x"], sb["x"])
            gap = max(sb["y"] - (sa["y"] + sa["h"]), sa["y"] - (sb["y"] + sb["h"]))
            if ov >= 0.5 * min(sa["w"], sb["w"]) and gap < 0.06:
                parent[find(a)] = find(b)
    out = {}
    for i in idx:
        out.setdefault(find(i), []).append(i)
    for gid, (_, members) in enumerate(sorted(out.items())):
        for i in members:
            slots[i]["group"] = gid
    return [sorted(m) for _, m in sorted(out.items())]


def raw_features(entry) -> tuple[float, ...]:
    texts = [s for s in entry["slots"] if s["cls"] == "text"]
    fonts = [s.get("font", 0.0) for s in texts if s.get("font")]
    return (math.log(entry["ratio"]), len(texts), entry["role_counts"]["action"], entry["role_counts"]["visual"],
            sum(s["w"] * s["h"] for s in texts), max(fonts, default=0.0),
            (max(fonts) / max(1e-6, min(fonts))) if fonts else 1.0, len(entry["groups"]),
            math.log1p(sum(s.get("chars", 0) for s in texts)))


def _signature(entry) -> tuple:
    q = lambda v: round(v * 10)
    return (round(math.log(entry["ratio"]) * 10),
            tuple(sorted((s["cls"], q(s["x"]), q(s["y"]), q(s["w"]), q(s["h"])) for s in entry["slots"])))


def entries_hash(entries) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assemble(entries: Iterable[dict], *, held_out_families: Iterable[str], held_out_source_ids: Iterable[str],
             build: dict, split_manifest: dict | None = None) -> dict:
    """Train-only, near-duplicate-free library with training-partition statistics."""
    held_f, held_s = set(held_out_families), set(held_out_source_ids)
    kept, seen, stats = [], set(), {"held_out": 0, "validation": 0, "near_duplicates": 0}
    for e in entries:
        if split_manifest is not None:
            from training.families import split_of as frozen_split
            try:
                e = {**e, 'split': frozen_split(e['source_family'], split_manifest)}
            except ValueError:
                stats['held_out'] += 1
                continue
        if e["source_family"] in held_f or e.get("source_id") in held_s:
            stats["held_out"] += 1
            continue
        if e["split"] != "train":
            stats["validation"] += 1
            continue
        sig = _signature(e)
        if sig in seen:
            stats["near_duplicates"] += 1
            continue
        seen.add(sig)
        kept.append(e)
    kept.sort(key=lambda e: e["layout_id"])
    feats = [e["features"] for e in kept]
    n = max(1, len(feats))
    mean = [sum(f[i] for f in feats) / n for i in range(len(FEATURE_NAMES))]
    std = [max(1e-2, math.sqrt(sum((f[i] - mean[i]) ** 2 for f in feats) / n)) for i in range(len(FEATURE_NAMES))]
    return {"schema": SCHEMA, "feature_names": list(FEATURE_NAMES),
            "normalization": {"partition": "train", "mean": [round(v, 6) for v in mean], "std": [round(v, 6) for v in std]},
            "held_out": {"families": sorted(held_f), "source_ids": sorted(held_s)},
            "build": {**build, **stats, "entries": len(kept)}, "embedding": None,
            "family_split_manifest": split_manifest,
            "entries_sha256": entries_hash(kept), "entries": kept}


ENTRY_KEYS = ("layout_id", "source_family", "split", "provenance", "ratio", "role_counts", "slots", "groups", "features")


def validate_library(lib: Any, *, held_out_families: Iterable[str] = (), held_out_source_ids: Iterable[str] = ()) -> dict:
    """Fail closed: schema, integrity hash, train-only split, held-out families,
    finite normalized geometry, and no unsupported embedding metadata."""
    if not isinstance(lib, dict) or lib.get("schema") != SCHEMA:
        raise LibraryError("incompatible_schema", str(lib.get("schema") if isinstance(lib, dict) else type(lib).__name__))
    if lib.get("feature_names") != list(FEATURE_NAMES):
        raise LibraryError("incompatible_features")
    emb = lib.get("embedding")
    if emb is not None and (not isinstance(emb, dict) or (emb.get("encoder"), emb.get("preprocessing")) not in SUPPORTED_EMBEDDINGS):
        raise LibraryError("incompatible_embedding", json.dumps(emb, sort_keys=True)[:120])
    norm = lib.get("normalization") or {}
    if norm.get("partition") != "train" or len(norm.get("mean", ())) != len(FEATURE_NAMES) or len(norm.get("std", ())) != len(FEATURE_NAMES):
        raise LibraryError("incompatible_normalization")
    for key in ('mean', 'std'):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or key == 'std' and v <= 0 for v in norm[key]):
            raise LibraryError('incompatible_normalization')
    entries = lib.get("entries")
    if not isinstance(entries, list) or not entries:
        raise LibraryError("empty_library")
    if entries_hash(entries) != lib.get("entries_sha256"):
        raise LibraryError("integrity_mismatch")
    held_f = set(held_out_families) | set((lib.get("held_out") or {}).get("families", ()))
    held_s = set(held_out_source_ids) | set(VISUAL_SUITE_SOURCE_IDS) | set((lib.get("held_out") or {}).get("source_ids", ()))
    for e in entries:
        if not isinstance(e, dict) or any(k not in e for k in ENTRY_KEYS):
            raise LibraryError("malformed_entry", str(e.get("layout_id") if isinstance(e, dict) else e)[:40])
        if e["split"] != "train":
            raise LibraryError("non_training_split", e["layout_id"])
        if lib.get('family_split_manifest') is not None:
            from training.families import split_of as frozen_split
            try:
                verified_split = frozen_split(e['source_family'], lib['family_split_manifest'])
            except (ValueError, TypeError, AttributeError) as exc:
                raise LibraryError('family_split_mismatch', str(exc)) from exc
            if verified_split != 'train':
                raise LibraryError('family_split_mismatch', e['layout_id'])
        if e["source_family"] in held_f or e.get("source_id") in held_s:
            raise LibraryError("held_out_family", e["layout_id"])
        r = e["ratio"]
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
            raise LibraryError("invalid_ratio", e["layout_id"])
        if not isinstance(e.get('features'), list) or len(e['features']) != len(FEATURE_NAMES) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in e['features']):
            raise LibraryError('incompatible_features', e['layout_id'])
        for s in e["slots"]:
            for k in ("x", "y", "w", "h"):
                v = s.get(k)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not -1e-6 <= v <= 1 + 1e-6:
                    raise LibraryError("invalid_slot", f"{e['layout_id']}.{k}")
            if s["x"] + s["w"] > 1.001 or s["y"] + s["h"] > 1.001 or s.get("cls") not in ("text", "action", "visual", "other"):
                raise LibraryError("invalid_slot", e["layout_id"])
            if any(k in s for k in ("content", "asset_id", "font_family", "color")):
                raise LibraryError("content_in_library", e["layout_id"])
    return lib


_CACHE: dict[str, tuple[float, dict]] = {}


def load_library(path: Path) -> dict:
    path = Path(path)
    stamp = path.stat().st_mtime
    hit = _CACHE.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        lib = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LibraryError("unreadable", str(exc)) from exc
    validate_library(lib)
    _CACHE[str(path)] = (stamp, lib)
    return lib
