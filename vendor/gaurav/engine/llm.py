"""Text-only language-model reading of the source JSON (workstream S6).

    from engine.llm import read
    r = read(source, analysis)          # Reading; empty (r.roles == {}) when nothing was learned

A small instruction-tuned LLM (Qwen3 / Gemma 3 GGUF through llama-cpp-python,
CPU only, no vision) gets a compact, id-free listing of the design's elements
(type, text, font-size rank, box in % of the canvas, z, colour, group letters,
"inside #k" hints) and answers three structured questions in one call:

  1. the role of every element (the analyzer's fine enum, restricted per
     element type by the grammar);
  2. semantic groups: sets of elements that must stay together in every
     layout, each with a one-word reason;
  3. a plan prior: archetype, focal element, the most important elements, and
     per target family (tall / square / wide / banner) the PlanOptions-level
     preferences the candidate generator already understands (hero side, chip
     placement, banner text flow, keep the source's structure or not).

The answer is grammar-constrained to a JSON schema (temperature 0, fixed
seed, no repeat penalty), so a wrong enum choice is the only failure mode.
`interpret()` maps it onto the analyzer's units the way engine/perceive.py
maps the VLM's answer: coarse classes with a *derived* confidence (BASE_CONF =
the bench accuracy per class, docs/LLM_BENCH.md), so `role_merge.apply_votes`
takes the reading as the third voter unchanged; the groups and the plan prior
are the two new signals (engine/resize.understand, engine/candidates).

Nothing here is a dependency. `read()` never raises; it returns an empty
Reading with a `status` when the switch is off (CHAMELEON_LLM must be
explicitly on: the default is off, and then llama_cpp is never imported),
when no model file is found, on a timeout, or when the answer is unusable,
and the engine proceeds with the rules. The per-task budget
CHAMELEON_LLM_BUDGET_S (default 8 s) covers model load + call; MAX_TIMEOUTS
timeouts or MAX_FAILURES backend errors switch the adapter off for the
process. Answers are cached by a hash of the source JSON (memory, plus
CHAMELEON_LLM_CACHE_DIR on disk), so the many targets of one source pay once.

Backends are pluggable (`set_backend`): tests use a stub and no weights; a
backend implements `complete(messages, schema, max_tokens, deadline) -> str`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from engine import perceive as P
from engine.analyze import BACKGROUND, Analysis
from engine.geometry import Box

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
# preferred model files under models/ (first existing wins; CHAMELEON_LLM_MODEL overrides);
# the order is the bench's verdict (docs/LLM_BENCH.md)
DEFAULT_MODELS = ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "Qwen3-1.7B-Q4_K_M.gguf", "gemma-3-4b-it-Q4_K_M.gguf")
PROMPT_VERSION = "s6.3"           # bump when the listing / prompt / schema change: it keys the cache

DEFAULT_BUDGET_S = 8.0
DEFAULT_SEED = 7
DEFAULT_N_CTX = 4096
MAX_THREADS = 8
MAX_TIMEOUTS = 2
MAX_FAILURES = 2
MIN_REMAINING_S = 1.0
CACHE_ENTRIES = 128
MAX_ELEMENTS = 45                 # listing cap: texts first, then images, then the largest shapes
MAX_GROUPS = 8
MAX_GROUP_MEMBERS = 8
MAX_IMPORTANCE = 5
TEXT_CHARS = 60                   # text content shown per element

TEXT_ROLES = ["headline", "subhead", "body", "cta", "badge", "fine_print"]
IMAGE_ROLES = ["logo", "hero", "image", "background", "decorative", "badge"]
SHAPE_ROLES = ["background", "panel", "decorative", "badge", "cta", "shape"]
GROUP_REASONS = ["button", "badge", "sticker", "lockup", "caption", "list", "frame", "logo", "contact", "other"]
ARCHETYPES = list(P.ARCHETYPES)
FOCALS = ["hero", "headline"]
FAMILIES = ("tall", "square", "wide", "banner")
HERO_SIDES = ["left", "right", "stack"]
BANNER_HERO_SIDES = ["left", "right"]
CHIPS = ["row", "corner", "beside_cta"]
BANNER_CHIPS = ["row", "corner", "stack"]
TEXT_FLOWS = ["stacked", "inline", "column"]

# derived confidence per coarse class (engine.perceive.COARSE): the chosen
# model's bench accuracy on the DesignAsCode-labelled designs with ids
# stripped (docs/LLM_BENCH.md), the same way perceive.BASE_CONF carries the
# VLM's. Until the bench ran these are the VLM's values.
BASE_CONF = {P.LOGO: 0.85, P.HEADLINE: 0.85, P.CTA: 0.8, P.PICTURE: 0.85, P.BADGE: 0.7, P.SECONDARY: 0.6,
             P.MINOR: 0.5}

SYSTEM_PROMPT = """You are a precise graphic-design analyst. You get the element list of a finished marketing design and answer with one JSON object, nothing else.

Element lines read "#k type ..." in reading order (top to bottom, left to right). Positions are percent of the canvas: x y = top-left corner, w h = size. z = stacking order (higher is drawn on top). A text line shows its text, font size in px and its size rank among the texts (rank1 = largest). "grp A" marks elements that already belong to one group; "in #k" means the element sits inside element #k.

Answer these three questions.

1. "roles": one role per element, keyed by its number.
Text roles: headline = the main message, the most prominent text; subhead = supporting text near the headline; body = longer descriptive copy; cta = a button label or call to action (Shop now, Learn more); badge = a short label, tag, sticker, price, date or stamp; fine_print = legal, contact or very small text.
Image roles: logo = the brand logo; hero = the main product photo or key picture; image = another, secondary picture; background = a picture filling the canvas behind everything; decorative = an ornament; badge = a sticker or icon.
Shape roles: background = fills the canvas; panel = a large flat colour area behind content; decorative = an ornament (line, blob, faint or bleeding shape); badge or cta = the pill behind a badge or button label; shape = anything else.
Exactly one headline; at most one hero, one logo and one cta.

2. "groups": up to 8 sets of elements that must stay together as one piece in every layout (a label with its pill, a date block, an icon with its caption, a photo with its frame, a logo with its tagline), each with a one-word reason: button | badge | sticker | lockup | caption | list | frame | logo | contact | other. Do not repeat the grp groups; use [] when there are none.

3. The plan for other canvas sizes.
"archetype": product_ad (a product or offer with a photo and a call to action) | event_poster (an event with date / place details) | editorial (mostly text, article or list-like) | quote (a saying as the main content) | grid (several equal items in rows or a grid) | other.
"focal": what the eye should land on first: hero (the picture) or headline (the text).
"importance": the numbers of the 1-5 most important elements, most important first.
"plans": for each target shape, how the design should be laid out.
tall (portrait, e.g. 1080x1920 or a 160x600 strip): chips = where badges and stickers go: row (one row in the column) | corner (the canvas corners) | beside_cta (next to the button); keep = true if the source's own composition (its bands, split or stack) should be kept rather than rebuilt from a template.
square (1080x1080): hero = left | right (the picture beside the text) | stack (above or below the text); chips; keep.
wide (1200x628, 1600x900): hero = left | right | stack; chips; keep.
banner (728x90, 1584x396): hero = left | right (which end of the strip); text = stacked (headline above the rest) | inline (headline and subhead side by side) | column (a text column beside the picture); chips = row | corner | stack; keep."""


class Backend(Protocol):
    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int,
                 deadline: float) -> str: ...


_state: dict[str, Any] = {
    "backend": None, "load_error": None, "disabled": None, "timeouts": 0, "failures": 0,
    "load_s": None, "cache": {}, "cache_order": [], "last": {}, "calls": 0,
}


@dataclass
class Reading:
    """What the model said, mapped onto the analyzer's units and elements."""

    roles: dict[str, tuple[str, float]] = field(default_factory=dict)   # unit key -> (coarse role, confidence)
    fine: dict[str, str] = field(default_factory=dict)                  # unit key -> fine role
    element_roles: dict[str, str] = field(default_factory=dict)         # element id -> fine role
    groups: list[tuple[list[str], str]] = field(default_factory=list)   # (element ids, reason)
    archetype: str | None = None
    focal: str | None = None
    importance: list[str] = field(default_factory=list)                 # unit keys, most important first
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)      # family -> {hero, chips, text, keep}
    status: str = "off"               # ok | cached | off | no_model | budget | no_answer | error
    elapsed: float = 0.0
    raw: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return bool(self.roles or self.groups or self.plans)

    def get(self, key: str) -> tuple[str, float] | None:
        return self.roles.get(key)

    def perception(self) -> P.Perception:
        """The reading as the VLM-shaped vote role_merge.apply_votes takes."""
        return P.Perception(roles=dict(self.roles), fine=dict(self.fine), archetype=self.archetype,
                            importance=list(self.importance), status=self.status, elapsed=self.elapsed)

    def as_dict(self) -> dict[str, Any]:
        return {"roles": {k: [r, round(c, 3)] for k, (r, c) in self.roles.items()}, "fine": dict(self.fine),
                "element_roles": dict(self.element_roles), "groups": [[list(ids), why] for ids, why in self.groups],
                "archetype": self.archetype, "focal": self.focal, "importance": list(self.importance),
                "plans": json.loads(json.dumps(self.plans)), "status": self.status,
                "elapsed": round(self.elapsed, 3), "raw": self.raw}

    @classmethod
    def from_dict(cls, d: dict[str, Any], status: str | None = None) -> "Reading":
        return cls(roles={k: (v[0], float(v[1])) for k, v in (d.get("roles") or {}).items()},
                   fine=dict(d.get("fine") or {}), element_roles=dict(d.get("element_roles") or {}),
                   groups=[(list(ids), str(why)) for ids, why in (d.get("groups") or [])],
                   archetype=d.get("archetype"), focal=d.get("focal"), importance=list(d.get("importance") or []),
                   plans=dict(d.get("plans") or {}), status=status or d.get("status", "ok"),
                   elapsed=float(d.get("elapsed", 0.0)), raw=d.get("raw"))


# ---------------------------------------------------------------------------
# switches


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """The reading runs only when CHAMELEON_LLM is explicitly on (default off)."""
    return _truthy(os.environ.get("CHAMELEON_LLM", ""))


def budget_s() -> float:
    v = os.environ.get("CHAMELEON_LLM_BUDGET_S", "").strip()
    try:
        return max(0.0, float(v)) if v else DEFAULT_BUDGET_S
    except ValueError:
        return DEFAULT_BUDGET_S


def model_path() -> Path | None:
    m = os.environ.get("CHAMELEON_LLM_MODEL", "").strip()
    if m:
        return Path(m)
    root = Path(os.environ.get("CHAMELEON_LLM_DIR", "") or MODELS_DIR)
    for name in DEFAULT_MODELS:
        if (root / name).is_file():
            return root / name
    return None


def available() -> bool:
    if not enabled() or _state["disabled"]:
        return False
    if _state["backend"] is not None:
        return True
    if _state["load_error"]:
        return False
    p = model_path()
    return bool(p and p.is_file())


def status() -> str:
    if not enabled():
        return "off: CHAMELEON_LLM is not on"
    if _state["disabled"]:
        return "off: " + _state["disabled"]
    if _state["load_error"]:
        return "off: " + _state["load_error"]
    if _state["backend"] is not None:
        return "on (loaded in %.1fs)" % (_state["load_s"] or 0.0)
    p = model_path()
    if not p or not p.is_file():
        return "off: no model (set CHAMELEON_LLM_MODEL or put a GGUF in models/)"
    return "ready: " + p.name


def stats() -> dict[str, Any]:
    return {"load_s": _state["load_s"], "calls": _state["calls"], "last": dict(_state["last"]),
            "timeouts": _state["timeouts"], "failures": _state["failures"], "status": status()}


def set_backend(backend: Optional[Backend]) -> None:
    _state["backend"] = backend
    _state["load_error"] = None


def _reset_for_tests() -> None:
    _state.update({"backend": None, "load_error": None, "disabled": None, "timeouts": 0, "failures": 0,
                   "load_s": None, "cache": {}, "cache_order": [], "last": {}, "calls": 0})


def _disable(reason: str) -> None:
    if not _state["disabled"]:
        _state["disabled"] = reason
        logger.warning("llm switched off: %s", reason)


# ---------------------------------------------------------------------------
# the listing (id-free, canonical order)


def _hex(color: Any) -> str | None:
    """rgba(…) / #rrggbb -> '#RRGGBB' (+ ' a0.5' when translucent)."""
    if not isinstance(color, str):
        return None
    s = color.strip()
    if s.startswith("#") and len(s) in (7, 9):
        return s[:7].upper()
    inner = s[s.find("(") + 1: s.rfind(")")]
    parts = [p.strip() for p in inner.split(",")]
    try:
        r, g, b = (max(0, min(255, int(float(p)))) for p in parts[:3])
    except (ValueError, IndexError):
        return None
    out = "#%02X%02X%02X" % (r, g, b)
    if len(parts) > 3:
        try:
            a = float(parts[3])
            if a < 0.99:
                out += " a%.1f" % a
        except ValueError:
            pass
    return out


def _pct(v: float, span: float) -> int:
    return int(round(100.0 * v / span)) if span > 0 else 0


def _clean_text(s: Any) -> str:
    t = " ".join(str(s or "").split())
    return t if len(t) <= TEXT_CHARS else t[:TEXT_CHARS - 1] + "…"


def listing(source: dict[str, Any]) -> tuple[str, list[str]]:
    """The element list the model reads and, in the same order, the ids
    behind its 1-based indices. Reading order (y, then x), scale-free, no
    ids; at most MAX_ELEMENTS elements (texts, then images, then the largest
    shapes)."""
    W = float(source["canvas"]["width"])
    H = float(source["canvas"]["height"])
    elements = list(source.get("elements") or [])
    z_of = {id(el): float(el.get("z_index", i) if el.get("z_index") is not None else i)
            for i, el in enumerate(elements)}

    def canon(el: dict[str, Any]) -> tuple:
        b = Box.of(el)
        return (round(b.y / H, 2) if H else 0.0, round(b.x / W, 2) if W else 0.0, z_of[id(el)],
                -round(b.area, 1), str(el.get("type")), _clean_text(el.get("content")))

    ordered = sorted(elements, key=canon)
    if len(ordered) > MAX_ELEMENTS:
        prio = {"text": 0, "image": 1}
        keep = sorted(ordered, key=lambda el: (prio.get(str(el.get("type")), 2), -Box.of(el).area))[:MAX_ELEMENTS]
        keep_ids = {id(el) for el in keep}
        ordered = [el for el in ordered if id(el) in keep_ids]
    index = {id(el): k for k, el in enumerate(ordered, 1)}
    boxes = {id(el): Box.of(el) for el in ordered}
    texts = [el for el in ordered if el.get("type") == "text"]
    sizes = sorted({float(el.get("font_size", 0.0) or 0.0) for el in texts}, reverse=True)
    rank = {s: i + 1 for i, s in enumerate(sizes)}
    letters: dict[str, str] = {}
    for el in ordered:
        gid = el.get("group_id")
        if gid and str(gid) not in letters:
            letters[str(gid)] = chr(ord("A") + len(letters)) if len(letters) < 26 else "G%d" % len(letters)
    # holder hints: the smallest flat shape / image drawn below the element
    # that holds >= 80 % of it and is under a quarter of the canvas
    holders = [el for el in ordered if el.get("type") in ("shape", "image")
               and 0 < boxes[id(el)].area < 0.25 * W * H
               and str(el.get("shape_type", "rectangle") or "rectangle") != "line"]
    orient = "square" if abs(W - H) <= 0.02 * max(W, H) else ("portrait" if H > W else "landscape")
    if W >= 3.0 * H:
        orient = "wide strip"
    elif H >= 3.0 * W:
        orient = "tall strip"
    n_t = len(texts)
    n_i = sum(1 for el in ordered if el.get("type") == "image")
    n_s = len(ordered) - n_t - n_i
    lines = ["Canvas %dx%d px (%s). %d elements: %d text, %d image, %d shape." % (W, H, orient, len(ordered), n_t, n_i, n_s)]
    for el in ordered:
        b = boxes[id(el)]
        t = str(el.get("type"))
        parts = ["#%d" % index[id(el)], t]
        if t == "text":
            fs = float(el.get("font_size", 0.0) or 0.0)
            parts.append('"%s"' % _clean_text(el.get("content")))
            parts.append("%gpx" % round(fs, 1))
            parts.append("rank%d/%d" % (rank.get(fs, len(sizes)), len(sizes)))
            if str(el.get("font_weight", "")).lower() in ("bold", "700", "800", "900"):
                parts.append("bold")
            if str(el.get("font_style", "")).lower() == "italic":
                parts.append("italic")
            fam = str(el.get("font_family", "") or "").lower()
            if "serif" in fam and "sans" not in fam:
                parts.append("serif")
            c = _hex(el.get("color"))
            if c:
                parts.append(c)
            n_lines = str(el.get("content") or "").count("\n") + 1
            if n_lines > 1:
                parts.append("%dlines" % n_lines)
        elif t == "image":
            parts.append(str(el.get("fit", "contain") or "contain"))
        else:
            st = str(el.get("shape_type", "rectangle") or "rectangle")
            parts.append({"rectangle": "rect"}.get(st, st))
            c = _hex(el.get("fill_color"))
            if c:
                parts.append("fill " + c)
            if float(el.get("corner_radius", 0.0) or 0.0) > 0:
                parts.append("rounded")
        parts.append("x%d y%d w%d h%d z%g" % (_pct(b.x, W), _pct(b.y, H), _pct(b.w, W), _pct(b.h, H), z_of[id(el)]))
        rot = float(el.get("rotation", 0.0) or 0.0)
        if rot:
            parts.append("rot%g" % round(rot, 1))
        op = float(el.get("opacity", 1.0) if el.get("opacity") is not None else 1.0)
        if op < 0.99:
            parts.append("op%.2g" % op)
        gid = el.get("group_id")
        if gid:
            parts.append("grp " + letters[str(gid)])
        if t != "shape" or b.area < 0.02 * W * H:
            best = None
            for h in holders:
                if h is el or z_of[id(h)] >= z_of[id(el)]:
                    continue
                hb = boxes[id(h)]
                if hb.area <= b.area or hb.contains_fraction(b) < 0.8:
                    continue
                if best is None or hb.area < boxes[id(best)].area:
                    best = h
            if best is not None:
                parts.append("in #%d" % index[id(best)])
        lines.append(" ".join(parts))
    return "\n".join(lines), [str(el.get("id")) for el in ordered]


def schema_for(source: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    """The answer's JSON schema for this listing: roles keyed by the element
    index (the grammar spells each key before the model picks the role, so a
    small model never loses its place), then groups, importance, archetype,
    focal and the per-family plans. Property order is the answer order."""
    by_id = {str(el.get("id")): el for el in source.get("elements") or []}
    role_props = {}
    for k, eid in enumerate(ids, 1):
        t = str(by_id.get(eid, {}).get("type", "shape"))
        role_props[str(k)] = {"type": "string", "enum": TEXT_ROLES if t == "text" else IMAGE_ROLES if t == "image" else SHAPE_ROLES}
    n = len(ids)
    # indices as string enums: the grammar then guarantees every index names
    # a listed element (llama.cpp does not enforce integer minimum / maximum)
    idx = {"type": "string", "enum": [str(k) for k in range(1, n + 1)]}
    fam = {
        "tall": {"chips": CHIPS, "keep": None},
        "square": {"hero": HERO_SIDES, "chips": CHIPS, "keep": None},
        "wide": {"hero": HERO_SIDES, "chips": CHIPS, "keep": None},
        "banner": {"hero": BANNER_HERO_SIDES, "text": TEXT_FLOWS, "chips": BANNER_CHIPS, "keep": None},
    }
    plans = {}
    for f, keys in fam.items():
        props = {k: ({"type": "boolean"} if v is None else {"type": "string", "enum": v}) for k, v in keys.items()}
        plans[f] = {"type": "object", "properties": props, "required": list(keys), "additionalProperties": False}
    return {
        "type": "object",
        "properties": {
            "roles": {"type": "object", "properties": role_props, "required": list(role_props), "additionalProperties": False},
            "groups": {"type": "array", "maxItems": MAX_GROUPS, "items": {
                "type": "object",
                "properties": {"m": {"type": "array", "items": idx, "minItems": 2, "maxItems": MAX_GROUP_MEMBERS},
                               "why": {"type": "string", "enum": GROUP_REASONS}},
                "required": ["m", "why"], "additionalProperties": False}},
            "importance": {"type": "array", "items": idx, "minItems": 1, "maxItems": min(MAX_IMPORTANCE, max(1, n))},
            "archetype": {"type": "string", "enum": ARCHETYPES},
            "focal": {"type": "string", "enum": FOCALS},
            "plans": {"type": "object", "properties": plans, "required": list(fam), "additionalProperties": False},
        },
        "required": ["roles", "groups", "importance", "archetype", "focal", "plans"],
        "additionalProperties": False,
    }


def messages_for(text: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Design:\n" + text + "\n\nJSON:"}]


def max_tokens_for(n: int) -> int:
    return min(900, 300 + 11 * n)


# ---------------------------------------------------------------------------
# interpretation


def _font(el: dict[str, Any]) -> float:
    return float(el.get("font_size", 0.0) or 0.0)


def interpret(answer: dict[str, Any], ids: list[str], source: dict[str, Any], analysis: Analysis) -> Reading:
    """Map the model's answer onto elements and units: coarse unit roles with
    derived confidence, fine roles, groups (ids), archetype, focal,
    importance (unit keys) and plans. Tolerant: whatever does not fit the
    design is dropped, never raised."""
    by_id = {str(el.get("id")): el for el in source.get("elements") or []}
    n = len(ids)
    raw_roles = answer.get("roles") if isinstance(answer.get("roles"), dict) else {}
    element_roles: dict[str, str] = {}
    for k, eid in enumerate(ids, 1):
        role = raw_roles.get(str(k))
        t = str(by_id.get(eid, {}).get("type", ""))
        allowed = TEXT_ROLES if t == "text" else IMAGE_ROLES if t == "image" else SHAPE_ROLES
        if role in allowed:
            element_roles[eid] = role

    def idx_ok(v: Any) -> bool:
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= n

    def idx(v: Any) -> int:
        return int(v)

    # units: the role of the members that name the unit (majority, then the biggest font / area)
    fine: dict[str, str] = {}
    coarse: dict[str, str] = {}
    for u in analysis.units:
        if u.role == BACKGROUND:
            continue
        fam = P.unit_family(u, by_id)
        members = [i for i in u.ids if by_id.get(i, {}).get("type") == fam] or list(u.ids)
        votes: dict[str, float] = {}
        for i in members:
            r = element_roles.get(i)
            if r is None:
                continue
            el = by_id.get(i, {})
            w = 1.0 + (_font(el) if fam == "text" else Box.of(el).area) * 1e-6
            votes[r] = votes.get(r, 0.0) + w
        if not votes:
            continue
        role = max(votes.items(), key=lambda t: t[1])[0]
        c = P.coarse_of(role)
        if c not in P.ALLOWED[fam]:
            continue
        fine[u.key] = role
        coarse[u.key] = c
    order: list[str] = []
    for v in answer.get("importance") or []:
        if idx_ok(v):
            key = analysis.element_unit.get(ids[idx(v) - 1])
            if key and key not in order:
                order.append(key)
    roles = {k: (c, _confidence(k, c, coarse, order)) for k, c in coarse.items()}

    groups: list[tuple[list[str], str]] = []
    seen: set[tuple[str, ...]] = set()
    schema_groups = set()
    for el in by_id.values():
        if el.get("group_id"):
            schema_groups.add(str(el.get("group_id")))
    for g in answer.get("groups") or []:
        if not isinstance(g, dict):
            continue
        mem = []
        for v in g.get("m") or []:
            if idx_ok(v) and ids[idx(v) - 1] not in mem:
                mem.append(ids[idx(v) - 1])
        if len(mem) < 2:
            continue
        key = tuple(sorted(mem))
        if key in seen:
            continue
        seen.add(key)
        why = g.get("why") if g.get("why") in GROUP_REASONS else "other"
        groups.append((mem, why))
    plans: dict[str, dict[str, Any]] = {}
    raw_plans = answer.get("plans") if isinstance(answer.get("plans"), dict) else {}
    for f in FAMILIES:
        p = raw_plans.get(f)
        if not isinstance(p, dict):
            continue
        clean: dict[str, Any] = {}
        hero_ok = BANNER_HERO_SIDES if f == "banner" else HERO_SIDES
        chips_ok = BANNER_CHIPS if f == "banner" else CHIPS
        if p.get("hero") in hero_ok:
            clean["hero"] = p["hero"]
        if p.get("chips") in chips_ok:
            clean["chips"] = p["chips"]
        if f == "banner" and p.get("text") in TEXT_FLOWS:
            clean["text"] = p["text"]
        if isinstance(p.get("keep"), bool):
            clean["keep"] = p["keep"]
        if clean:
            plans[f] = clean
    archetype = answer.get("archetype") if answer.get("archetype") in ARCHETYPES else None
    focal = answer.get("focal") if answer.get("focal") in FOCALS else None
    return Reading(roles=roles, fine=fine, element_roles=element_roles, groups=groups, archetype=archetype,
                   focal=focal, importance=order, plans=plans, status="ok", raw=answer)


def _confidence(key: str, coarse: str, all_coarse: dict[str, str], order: list[str]) -> float:
    """perceive.confidence with this module's BASE_CONF."""
    base = BASE_CONF.get(coarse, 0.5)
    if coarse not in P.UNIQUE:
        return base
    same = [k for k, c in all_coarse.items() if c == coarse]
    if len(same) <= 1:
        return base
    ranked = [k for k in order if k in same]
    if ranked and ranked[0] == key:
        return base
    return base / len(same)


# ---------------------------------------------------------------------------
# llama.cpp backend (text only)


def _default_threads() -> int:
    v = os.environ.get("CHAMELEON_LLM_THREADS", "").strip()
    if v.isdigit() and int(v) > 0:
        return int(v)
    return max(1, min(MAX_THREADS, os.cpu_count() or 1))


class LlamaCppBackend:
    """A GGUF chat model through llama-cpp-python: the GGUF's own chat template
    (rendered with jinja2, thinking switched off), grammar-constrained JSON,
    greedy decoding, streamed so the deadline can stop it."""

    def __init__(self, path: Path, n_threads: int | None = None, n_ctx: int = DEFAULT_N_CTX, seed: int = DEFAULT_SEED):
        from llama_cpp import Llama

        self.model_path = Path(path)
        self.seed = seed
        self.n_threads = n_threads or _default_threads()
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_batch=min(512, n_ctx), n_threads=self.n_threads,
                         n_threads_batch=self.n_threads, n_gpu_layers=0, seed=seed, logits_all=False, verbose=False)
        self.template = (self.llm.metadata or {}).get("tokenizer.chat_template") or ""
        self._env = None
        self._grammars: dict[str, Any] = {}
        self.last: dict[str, Any] = {}

    def render(self, messages: list[dict[str, str]]) -> str:
        """The prompt string for `messages` (+ the assistant header)."""
        if self.template:
            try:
                import jinja2

                if self._env is None:
                    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                             extensions=[jinja2.ext.loopcontrols])
                    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
                    self._env = env.from_string(self.template)
                return self._env.render(messages=messages, add_generation_prompt=True, enable_thinking=False,
                                        bos_token="", eos_token="",
                                        raise_exception=lambda m: (_ for _ in ()).throw(RuntimeError(m)))
            except Exception as exc:  # noqa: BLE001 - fall back to ChatML
                logger.info("chat template failed (%s); using ChatML", exc)
        out = []
        for m in messages:
            out.append("<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"]))
        out.append("<|im_start|>assistant\n")
        return "".join(out)

    def grammar(self, schema: dict[str, Any]):
        from llama_cpp import LlamaGrammar

        key = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        g = self._grammars.get(key)
        if g is None:
            g = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
            if len(self._grammars) > 64:
                self._grammars.clear()
            self._grammars[key] = g
        return g

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int, deadline: float) -> str:
        prompt = self.render(messages)
        t0 = time.perf_counter()
        n_prompt = len(self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))
        stream = self.llm.create_completion(
            prompt=prompt, grammar=self.grammar(schema), temperature=0.0, top_k=1, top_p=1.0, min_p=0.0,
            repeat_penalty=1.0, seed=self.seed, max_tokens=max_tokens, stream=True,
            stop=["<|im_end|>", "<end_of_turn>"])
        pieces: list[str] = []
        n_out = 0
        first = None
        stopped = False
        try:
            for chunk in stream:
                now = time.perf_counter()
                if first is None:
                    first = now
                piece = chunk["choices"][0].get("text")
                if piece:
                    pieces.append(piece)
                n_out += 1
                if now > deadline:
                    stopped = True
                    break
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        end = time.perf_counter()
        self.last = {"prefill_s": (first or end) - t0, "gen_s": end - (first or end), "gen_tokens": n_out,
                     "prompt_tokens": n_prompt, "stopped_at_deadline": stopped}
        return "".join(pieces)


def load(force: bool = False) -> Backend | None:
    """Load the configured model once. None when off, missing or broken."""
    if not enabled() or _state["disabled"]:
        return None
    if _state["backend"] is not None and not force:
        return _state["backend"]
    if _state["load_error"] and not force:
        return None
    p = model_path()
    if not p or not p.is_file():
        _state["load_error"] = "no model file"
        return None
    t0 = time.perf_counter()
    try:
        n_ctx = int(os.environ.get("CHAMELEON_LLM_CTX", "") or DEFAULT_N_CTX)
        seed = int(os.environ.get("CHAMELEON_LLM_SEED", "") or DEFAULT_SEED)
        backend = LlamaCppBackend(p, n_ctx=n_ctx, seed=seed)
    except Exception as exc:  # pragma: no cover - needs a broken install
        _state["load_error"] = "load failed: %s" % exc
        logger.warning("llm unavailable: %s", exc)
        return None
    _state["load_s"] = time.perf_counter() - t0
    _state["backend"] = backend
    _state["load_error"] = None
    logger.info("llm loaded %s in %.1fs", p.name, _state["load_s"])
    return backend


# ---------------------------------------------------------------------------
# cache


def source_hash(source: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(source, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _backend_tag() -> str:
    b = _state["backend"]
    if b is None:
        p = model_path()
        return p.name if p else "none"
    return getattr(getattr(b, "model_path", None), "name", None) or type(b).__name__


def cache_key(source: dict[str, Any]) -> str:
    h = hashlib.sha1()
    h.update(source_hash(source).encode())
    h.update(PROMPT_VERSION.encode())
    h.update(_backend_tag().encode())
    return h.hexdigest()


def _cache_dir() -> Path | None:
    d = os.environ.get("CHAMELEON_LLM_CACHE_DIR", "").strip()
    return Path(d) if d else None


def _cache_get(key: str) -> dict[str, Any] | None:
    if key in _state["cache"]:
        return _state["cache"][key]
    d = _cache_dir()
    if d:
        f = d / ("llm_%s.json" % key)
        if f.is_file():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
    return None


def _cache_put(key: str, value: dict[str, Any]) -> None:
    cache, order = _state["cache"], _state["cache_order"]
    if key not in cache:
        order.append(key)
        while len(order) > CACHE_ENTRIES:
            cache.pop(order.pop(0), None)
    cache[key] = value
    d = _cache_dir()
    if d:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / ("llm_%s.json" % key)).write_text(json.dumps(value))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# the call


def read(source: dict[str, Any], analysis: Analysis, budget: float | None = None,
         use_cache: bool = True) -> Reading:
    """Ask the model about the source. Never raises; an empty Reading (with a
    `status` saying why) means "no opinion"."""
    t0 = time.perf_counter()
    try:
        if not enabled():
            return Reading(status="off")
        budget = budget_s() if budget is None else float(budget)
        key = cache_key(source)
        if use_cache:
            hit = _cache_get(key)
            if hit is not None:
                r = Reading.from_dict(hit, status="cached")
                r.elapsed = time.perf_counter() - t0
                _state["last"] = {"cached": True}
                return r
        if _state["disabled"]:
            return Reading(status="off")
        backend = load()
        if backend is None:
            return Reading(status="no_model")
        text, ids = listing(source)
        if not ids:
            return Reading(status="no_answer")
        schema = schema_for(source, ids)
        remaining = budget - (time.perf_counter() - t0)
        if remaining < MIN_REMAINING_S:
            logger.info("llm reading skipped: %.1fs of the %.1fs budget left", remaining, budget)
            return Reading(status="budget", elapsed=time.perf_counter() - t0)
        deadline = time.perf_counter() + remaining
        t1 = time.perf_counter()
        try:
            out = backend.complete(messages_for(text), schema, max_tokens_for(len(ids)), deadline)
        except Exception as exc:  # noqa: BLE001
            _state["failures"] += 1
            logger.warning("llm call failed (%s)", exc)
            if _state["failures"] >= MAX_FAILURES:
                _disable("backend raised %d times" % _state["failures"])
            return Reading(status="error", elapsed=time.perf_counter() - t0)
        dt = time.perf_counter() - t1
        last = dict(getattr(backend, "last", {}) or {})
        last.update({"total_s": dt, "cached": False, "n_elements": len(ids)})
        _state["last"] = last
        _state["calls"] += 1
        if dt > remaining and not last.get("stopped_at_deadline"):
            last["stopped_at_deadline"] = True
        if last.get("stopped_at_deadline"):
            _state["timeouts"] += 1
            logger.warning("llm call hit its %.1fs deadline (%.1fs)", remaining, dt)
            if _state["timeouts"] >= MAX_TIMEOUTS:
                _disable("%d calls timed out" % _state["timeouts"])
        from engine.vlm import parse_json, validate

        answer = parse_json(out)
        if answer is None or validate(answer, schema):
            if answer is not None:
                logger.info("llm answer fails the schema: %s", "; ".join(validate(answer, schema)[:4]))
            return Reading(status="no_answer" if not last.get("stopped_at_deadline") else "budget",
                           elapsed=time.perf_counter() - t0)
        r = interpret(answer, ids, source, analysis)
        r.elapsed = time.perf_counter() - t0
        if use_cache:
            _cache_put(key, r.as_dict())
        return r
    except Exception as exc:  # never into the engine
        logger.warning("llm reading failed (%s)", exc)
        return Reading(status="error", elapsed=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# python -m engine.llm [task.json ...]: show the listing and the answer


def _main(paths: list[str]) -> int:  # pragma: no cover
    import glob

    from engine.analyze import analyze

    files = [p for pat in (paths or ["input/*.json"]) for p in sorted(glob.glob(pat))]
    os.environ.setdefault("CHAMELEON_LLM", "1")
    print("llm:", status())
    for f in files:
        task = json.loads(Path(f).read_text())
        src = task.get("source", task)
        a = analyze(src)
        text, ids = listing(src)
        print("\n=== %s\n%s" % (Path(f).name, text))
        r = read(src, a, budget=120.0, use_cache=False)
        print("-> %s in %.1fs (%s)" % (r.status, r.elapsed, json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                                                   for k, v in _state["last"].items()})))
        if r.raw:
            print(json.dumps(r.raw))
        for u in a.units:
            got = r.roles.get(u.key)
            print("  %-26s rules=%-11s llm=%s" % (u.key[:26], u.role, ("%s/%s %.2f" % (r.fine.get(u.key), got[0], got[1])) if got else "-"))
        print("  groups:", r.groups, "| archetype:", r.archetype, "focal:", r.focal, "| plans:", r.plans)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main(sys.argv[1:]))
