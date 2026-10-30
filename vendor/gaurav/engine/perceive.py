"""VLM perception of the source design (workstream S1): one vote among three.

    from engine.perceive import perceive
    p = perceive(source, analysis)          # Perception; p.roles == {} when nothing was learned

The source is rendered once (engine.render), one numbered magenta box is
drawn per analyzer unit (set of marks, the overlay that reached 0.75 coarse
agreement in docs/VLM_BENCH.md), and the local VLM (engine.vlm, Qwen3-VL
GGUF) is asked, with the *hinted* prompt (box number, element types, text),
for the role of every box, the design's archetype and the most important
boxes. The answer is grammar-constrained to the schema, so a wrong enum
choice is the only failure mode; the fine role is mapped onto the coarse
classes a layout engine can act on (`COARSE`), and a derived confidence is
attached (`confidence()`): the model gives no probabilities, so confidence is
the bench's accuracy on the role class, divided among boxes that share a
role that should be unique, and zero when the answer contradicts the box's
element type (an image cannot be a headline).

Nothing here is a dependency: `perceive()` never raises, returns an empty
Perception when the VLM is off (`CHAMELEON_VLM` must be explicitly on: the
default is off), has no weights, cannot render, times out or answers
nonsense, and the engine proceeds with the rules. A per-task time budget
(`CHAMELEON_VLM_BUDGET_S`, default 20 s) covers model load + render + call;
when it is exhausted perception is skipped. Results are cached by a hash of
the source JSON (memory, plus `CHAMELEON_VLM_CACHE_DIR` on disk), so the
many targets of one source pay for one call. `engine.vlm` is imported only
when the switch is on, so `CHAMELEON_VLM=0` never loads native code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.analyze import BACKGROUND, Analysis, Unit

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_S = 20.0
RENDER_SIZE = 896          # the overlay is drawn at this size ...
MODEL_SIDE = 448           # ... and sent to the model at this long side (docs/VLM_BENCH.md: 672 buys nothing)
MIN_REMAINING_S = 1.5      # less than this left of the budget: skip the call
CACHE_ENTRIES = 64
MAX_IMPORTANCE = 5

# fine roles the model chooses from (the analyzer's vocabulary minus background)
FINE_ROLES = ["logo", "hero", "image", "headline", "subhead", "body", "cta", "badge",
              "fine_print", "decorative", "panel", "shape"]
ROLE_HELP = {
    "logo": "brand logo",
    "hero": "the main product photo or key image",
    "image": "another, secondary photo or illustration",
    "headline": "the main message, the largest or most prominent text",
    "subhead": "supporting text under or near the headline",
    "body": "longer descriptive text",
    "cta": "a button or call-to-action label (e.g. Shop now, Learn more)",
    "badge": "a small sticker, pill, label, tag or ribbon (e.g. NEW, 50% OFF)",
    "fine_print": "legal, disclaimer or very small text",
    "decorative": "an ornamental shape with no content (circle, blob, line)",
    "panel": "a large flat colour area that sits behind other content",
    "shape": "any other shape",
}
ARCHETYPES = ["product_ad", "event_poster", "editorial", "quote", "grid", "other"]
ARCHETYPE_HELP = ("product_ad = a product or offer with a photo and a call to action; event_poster = an "
                  "event with date / place details; editorial = mostly text, article or list-like; quote = "
                  "a quotation or saying as the main content; grid = several equal items in rows or a "
                  "grid; other = none of these")

# coarse classes a layout engine acts on; the fine splits inside them stay with the rules
PICTURE, HEADLINE, CTA, LOGO, BADGE, SECONDARY, MINOR = ("picture", "headline", "cta", "logo", "badge",
                                                         "secondary", "minor")
COARSE = {"hero": PICTURE, "image": PICTURE, "headline": HEADLINE, "cta": CTA, "logo": LOGO, "badge": BADGE,
          "subhead": SECONDARY, "body": SECONDARY, "fine_print": SECONDARY, "text": SECONDARY,
          "decorative": MINOR, "panel": MINOR, "shape": MINOR, "background": MINOR}
UNIQUE = (HEADLINE, CTA, LOGO)           # one per design: boxes sharing one of these dilute each other
# derived confidence per coarse class: the 4B's hinted accuracy on the sample boxes
# whose ids give ground truth (docs/VLM_BENCH.md: 17/20) for the classes it is good at,
# lower where the bench showed it guessing (secondary vs badge, decoration vs panel)
BASE_CONF = {LOGO: 0.85, HEADLINE: 0.85, CTA: 0.8, PICTURE: 0.85, BADGE: 0.7, SECONDARY: 0.6, MINOR: 0.5}
# which coarse classes a box of this element family may be
ALLOWED = {"text": {HEADLINE, CTA, BADGE, SECONDARY}, "image": {LOGO, PICTURE, BADGE}, "shape": {MINOR, BADGE}}


def coarse_of(role: str | None) -> str:
    return COARSE.get(role or "", MINOR)


@dataclass
class Perception:
    roles: dict[str, tuple[str, float]] = field(default_factory=dict)   # unit key -> (coarse role, confidence)
    fine: dict[str, str] = field(default_factory=dict)                  # unit key -> the model's fine role
    archetype: str | None = None
    importance: list[str] = field(default_factory=list)                 # unit keys, most important first
    status: str = "off"               # ok | cached | off | no_model | no_render | budget | no_answer | error
    elapsed: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.roles)

    def get(self, key: str) -> tuple[str, float] | None:
        return self.roles.get(key)

    def as_dict(self) -> dict[str, Any]:
        return {"roles": {k: [r, round(c, 3)] for k, (r, c) in self.roles.items()}, "fine": dict(self.fine),
                "archetype": self.archetype, "importance": list(self.importance), "status": self.status,
                "elapsed": round(self.elapsed, 3)}

    @classmethod
    def from_dict(cls, d: dict[str, Any], status: str | None = None) -> "Perception":
        return cls(roles={k: (v[0], float(v[1])) for k, v in (d.get("roles") or {}).items()},
                   fine=dict(d.get("fine") or {}), archetype=d.get("archetype"),
                   importance=list(d.get("importance") or []), status=status or d.get("status", "ok"),
                   elapsed=float(d.get("elapsed", 0.0)))


# ---------------------------------------------------------------------------
# switches


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """Perception runs only when CHAMELEON_VLM is explicitly on (default off)."""
    return _truthy(os.environ.get("CHAMELEON_VLM", ""))


def budget_s() -> float:
    v = os.environ.get("CHAMELEON_VLM_BUDGET_S", "").strip()
    try:
        return max(0.0, float(v)) if v else DEFAULT_BUDGET_S
    except ValueError:
        return DEFAULT_BUDGET_S


# ---------------------------------------------------------------------------
# set of marks (shared with tools/vlm_bench.py)


def source_hash(source: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(source, sort_keys=True, separators=(",", ":"), default=str)
                        .encode("utf-8")).hexdigest()


def marks(analysis: Analysis) -> list[dict[str, Any]]:
    """One numbered mark per unit (groups as one box), reading order, backgrounds skipped."""
    units = [u for u in analysis.units if u.role != BACKGROUND]
    units.sort(key=lambda u: (round(u.box.y / 40.0), u.box.x))
    return [{"n": i, "key": u.key, "ids": list(u.ids), "role": u.role,
             "box": [u.box.x, u.box.y, u.box.w, u.box.h]} for i, u in enumerate(units, 1)]


def draw_marks(img, canvas_w: float, mk: list[dict[str, Any]]):
    from PIL import ImageDraw, ImageFont

    s = img.width / float(canvas_w)
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    size = max(14, int(img.width / 36))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    lw = max(2, int(img.width / 300))
    for m in mk:
        x, y, w, h = [v * s for v in m["box"]]
        d.rectangle([x, y, x + w, y + h], outline=(255, 0, 200), width=lw)
        label = str(m["n"])
        tw = d.textlength(label, font=font)
        tx, ty = max(0, x), max(0, y - size - 4)
        if ty < 0:
            ty = y
        d.rectangle([tx, ty, tx + tw + 8, ty + size + 4], fill=(255, 0, 200))
        d.text((tx + 4, ty + 1), label, fill=(255, 255, 255), font=font)
    return out


def unit_family(unit: Unit, by_id: dict[str, dict[str, Any]]) -> str:
    """text when the unit has a text member (a button is named by its label),
    else image when it has an image, else shape."""
    types = {str(by_id[i].get("type", "")) for i in unit.ids if i in by_id}
    if "text" in types:
        return "text"
    if "image" in types:
        return "image"
    return "shape"


def hints(source: dict[str, Any], mk: list[dict[str, Any]]) -> str:
    """What the engine knows about each box without looking: element types
    and the text they contain (hidden tasks give type and content too)."""
    by_id = {e["id"]: e for e in source["elements"]}
    lines = []
    for m in mk:
        kinds = []
        for i in m["ids"]:
            e = by_id.get(i)
            if e is None:
                continue
            if e["type"] == "text":
                t = " ".join(str(e.get("content", "")).split())
                kinds.append("text \"%s\"" % (t[:50] + ("..." if len(t) > 50 else "")))
            elif e["type"] == "image":
                kinds.append("image")
            else:
                kinds.append("shape")
        lines.append("%d: %s" % (m["n"], " + ".join(kinds)))
    return "\n".join(lines)


def prompt_and_schema(mk: list[dict[str, Any]], hint_text: str) -> tuple[str, dict[str, Any]]:
    n = len(mk)
    numbers = [str(m["n"]) for m in mk]
    roles_txt = "; ".join("%s = %s" % (r, ROLE_HELP[r]) for r in FINE_ROLES)
    prompt = (
        "This is a marketing design. %d numbered magenta boxes are drawn over its elements "
        "(the number tag sits at the top-left corner of each box). For every box number say what "
        "the element inside it is. Roles: %s. The boxes contain:\n%s\n"
        "Judge their role from how they look and where they sit. Then name the design's archetype "
        "(%s) and list the %d most important boxes, most important first. "
        "Reply with JSON only: {\"roles\": {\"1\": role, ..., \"%d\": role}, \"archetype\": archetype, "
        "\"importance\": [box numbers]}."
        % (n, roles_txt, hint_text, ARCHETYPE_HELP, min(MAX_IMPORTANCE, n), n)
    )
    schema = {
        "type": "object",
        "properties": {
            "roles": {
                "type": "object",
                "properties": {k: {"type": "string", "enum": FINE_ROLES} for k in numbers},
                "required": numbers,
                "additionalProperties": False,
            },
            "archetype": {"type": "string", "enum": ARCHETYPES},
            "importance": {"type": "array", "items": {"type": "string", "enum": numbers},
                           "minItems": 1, "maxItems": min(MAX_IMPORTANCE, n)},
        },
        "required": ["roles", "archetype", "importance"],
        "additionalProperties": False,
    }
    return prompt, schema


# ---------------------------------------------------------------------------
# interpretation


def interpret(answer: dict[str, Any], mk: list[dict[str, Any]], source: dict[str, Any],
              analysis: Analysis) -> Perception:
    """Map the model's answer onto units: coarse roles with derived
    confidence, the archetype and the importance ranking (unit keys)."""
    by_id = {e["id"]: e for e in source.get("elements", [])}
    units = {u.key: u for u in analysis.units}
    raw = answer.get("roles") or {}
    fine: dict[str, str] = {}
    coarse: dict[str, str] = {}
    for m in mk:
        role = raw.get(str(m["n"]))
        if role not in FINE_ROLES:
            continue
        unit = units.get(m["key"])
        if unit is None:
            continue
        c = coarse_of(role)
        if c not in ALLOWED[unit_family(unit, by_id)]:
            continue                     # contradicts the element type: no vote
        fine[m["key"]] = role
        coarse[m["key"]] = c
    order: list[str] = []
    for n in answer.get("importance") or []:
        key = next((m["key"] for m in mk if str(m["n"]) == str(n)), None)
        if key is not None and key not in order:
            order.append(key)
    roles = {k: (c, confidence(k, c, coarse, order)) for k, c in coarse.items()}
    archetype = answer.get("archetype")
    return Perception(roles=roles, fine=fine, archetype=archetype if archetype in ARCHETYPES else None,
                      importance=order, status="ok")


def confidence(key: str, coarse: str, all_coarse: dict[str, str], order: list[str]) -> float:
    """Derived confidence: the class's base accuracy, divided among the boxes
    that share a role that should be unique (the one the model ranked most
    important keeps the full value)."""
    base = BASE_CONF.get(coarse, 0.5)
    if coarse not in UNIQUE:
        return base
    same = [k for k, c in all_coarse.items() if c == coarse]
    if len(same) <= 1:
        return base
    ranked = [k for k in order if k in same]
    if ranked and ranked[0] == key:
        return base
    return base / len(same)


# ---------------------------------------------------------------------------
# the call


_cache: dict[str, dict[str, Any]] = {}
_cache_order: list[str] = []


def _cache_dir() -> Path | None:
    d = os.environ.get("CHAMELEON_VLM_CACHE_DIR", "").strip()
    return Path(d) if d else None


def _cache_get(key: str) -> dict[str, Any] | None:
    if key in _cache:
        return _cache[key]
    d = _cache_dir()
    if d:
        f = d / ("perceive_%s.json" % key)
        if f.is_file():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
    return None


def _cache_put(key: str, value: dict[str, Any]) -> None:
    if key not in _cache:
        _cache_order.append(key)
        while len(_cache_order) > CACHE_ENTRIES:
            _cache.pop(_cache_order.pop(0), None)
    _cache[key] = value
    d = _cache_dir()
    if d:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / ("perceive_%s.json" % key)).write_text(json.dumps(value))
        except Exception:
            pass


def _reset_for_tests() -> None:
    _cache.clear()
    _cache_order.clear()


def perceive(source: dict[str, Any], analysis: Analysis, budget: float | None = None,
             use_cache: bool = True) -> Perception:
    """Ask the VLM about the source. Never raises; an empty Perception (with
    a `status` saying why) means "no opinion"."""
    t0 = time.perf_counter()
    try:
        if not enabled():
            return Perception(status="off")
        budget = budget_s() if budget is None else float(budget)
        key = source_hash(source)
        if use_cache:
            hit = _cache_get(key)
            if hit is not None:
                p = Perception.from_dict(hit, status="cached")
                p.elapsed = time.perf_counter() - t0
                return p
        from engine import render as R
        from engine import vlm

        if not vlm.available():
            return Perception(status="no_model")
        mk = marks(analysis)
        if not mk:
            return Perception(status="no_answer")
        backend = vlm.load()
        if backend is None:
            return Perception(status="no_model")
        img = R.render_design(source, max_size=RENDER_SIZE)
        if img is None:
            return Perception(status="no_render")
        marked = draw_marks(img, float(source["canvas"]["width"]), mk)
        prompt, schema = prompt_and_schema(mk, hints(source, mk))
        remaining = budget - (time.perf_counter() - t0)
        if remaining < MIN_REMAINING_S:
            logger.info("perception skipped: %.1fs of the %.1fs budget left", remaining, budget)
            return Perception(status="budget", elapsed=time.perf_counter() - t0)
        answer = vlm.describe(marked, prompt, schema, max_tokens=48 + 8 * len(mk), timeout_s=remaining,
                              max_side=MODEL_SIDE, use_cache=use_cache)
        if not answer:
            return Perception(status="no_answer", elapsed=time.perf_counter() - t0)
        p = interpret(answer, mk, source, analysis)
        p.elapsed = time.perf_counter() - t0
        if use_cache:
            _cache_put(key, p.as_dict())
        return p
    except Exception as exc:  # never into the engine
        logger.warning("perception failed (%s)", exc)
        return Perception(status="error", elapsed=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# python -m engine.perceive [task.json ...]: show what the model sees


def _main(paths: list[str]) -> int:  # pragma: no cover
    import glob

    from engine.analyze import analyze

    files = [p for pat in (paths or ["input/*.json"]) for p in sorted(glob.glob(pat))]
    os.environ.setdefault("CHAMELEON_VLM", "1")
    for f in files:
        task = json.loads(Path(f).read_text())
        src = task.get("source", task)
        a = analyze(src)
        p = perceive(src, a, use_cache=False)
        print("%s: %s in %.1fs, archetype=%s" % (Path(f).name, p.status, p.elapsed, p.archetype))
        for m in marks(a):
            got = p.roles.get(m["key"])
            print("  %2d %-22s rules=%-11s vlm=%s" % (m["n"], m["key"][:22], m["role"],
                                                  ("%s/%s %.2f" % (p.fine.get(m["key"]), got[0], got[1])) if got else "-"))
        print("  importance:", p.importance)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main(sys.argv[1:]))
