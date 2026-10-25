"""Provider contract: typed proposals, isolated failures, strict validation.

A provider turns one immutable ProposalContext into candidate designs. It never
mutates the source, never imports the controller and never converts its own
failure into a fabricated candidate: failures are ProviderResult records.
Every proposal is validated against the source *before* materialization, so a
malformed optional provider can only remove its own proposals.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Protocol, runtime_checkable

from ..canonical import MUTABLE_FIELDS

CONTRACT_VERSION = "providers/1"
STATUSES = ("ok", "empty", "failed", "skipped")
# Rejection codes are stable identifiers (tests and traces match on them).
REJECTION_CODES = (
    "malformed", "canvas_mismatch", "unknown_id", "duplicate_id", "missing_id",
    "non_allowlisted_field", "boolean_number", "non_numeric", "non_finite",
    "negative_origin", "negative_size", "font_size_range", "element_order",
)
NEGATIVE_TOLERANCE = 0.5  # matches invariants.TOL


class ProposalRejected(ValueError):
    """Typed, inspectable rejection of one proposal before materialization."""

    def __init__(self, code: str, detail: str = "", element_id: str | None = None, field_name: str | None = None):
        assert code in REJECTION_CODES, code
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail, self.element_id, self.field = code, detail, element_id, field_name

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "element_id": self.element_id, "field": self.field}


@dataclass(frozen=True)
class Proposal:
    """One candidate layout. `design` is canvas + the original elements only."""

    provider: str
    template: str            # legacy pipeline template name (kept for rank/trace compatibility)
    name: str
    description: str
    design: dict
    family: str = ""          # structural family label: metadata, not diversity
    configuration_ids: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    def as_legacy(self) -> tuple[str, str, str, dict]:
        """The (template, name, description, design) tuple pipeline.generate builds."""
        return (self.template, self.name, self.description, self.design)


@dataclass
class ProviderResult:
    provider: str
    status: str
    proposals: tuple[Proposal, ...] = ()
    error: str | None = None
    elapsed_ms: float = 0.0
    rejections: tuple[dict, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "status": self.status, "proposals": len(self.proposals),
                "templates": [p.template for p in self.proposals], "error": self.error,
                "elapsed_ms": round(self.elapsed_ms, 1), "rejections": list(self.rejections),
                "diagnostics": self.diagnostics}


@dataclass(frozen=True)
class ProposalContext:
    """Immutable inputs shared by every provider for one resize task."""

    source: dict
    target: dict
    analysis: Any
    locks: dict
    profile: dict
    emphasis: str = "balanced"
    source_hash: str = ""
    role_status: dict = field(default_factory=dict)

    @property
    def same_size(self) -> bool:
        return self.source["canvas"] == self.target

    @property
    def graph(self):
        """C1's AnalysisGraph when present; None means use deterministic fallbacks."""
        return analysis_graph(self.source, self.analysis)

    def source_copy(self) -> dict:
        return copy.deepcopy(self.source)


@runtime_checkable
class Provider(Protocol):
    name: str
    cost: int  # relative scheduling cost; cheap providers run first

    def propose(self, context: ProposalContext, budget=None) -> ProviderResult: ...


def analysis_graph(source, analysis):
    """Guarded optional use of C1's relation graph (engine.relations)."""
    graph = getattr(analysis, "graph", None)
    if graph is not None:
        return graph
    try:  # pragma: no cover - depends on the C1 lane being integrated
        from .. import relations  # type: ignore
    except ImportError:
        return None
    try:  # pragma: no cover
        return relations.build_graph(source, analysis)
    except Exception:
        return None


def source_digest(source: dict) -> str:
    return hashlib.sha256(json.dumps(source, sort_keys=True, default=str).encode()).hexdigest()


def build_context(source, target, analysis=None, *, locks=None, options=None, emphasis="balanced") -> ProposalContext:
    """Exactly the analysis state pipeline.generate proposes from."""
    from ..analysis import analyze
    from ..learning import apply_roles
    options = dict(options or {})
    A = copy.deepcopy(analysis) if analysis else analyze(source, group_policies=options.get('group_policies'))
    role_status = apply_roles(source, A, enabled=options.get('role_mode', options.get("roles")))
    target = {"width": target["width"], "height": target["height"]}
    return ProposalContext(copy.deepcopy(source), target, A, dict(locks or {}), options, emphasis,
                           source_digest(source), role_status)


# ---------------------------------------------------------------- validation
def _number(value, eid, key):
    if isinstance(value, bool):
        raise ProposalRejected("boolean_number", f"{eid}.{key} is a boolean", eid, key)
    if not isinstance(value, (int, float)):
        raise ProposalRejected("non_numeric", f"{eid}.{key}={value!r}", eid, key)
    if not math.isfinite(value):
        raise ProposalRejected("non_finite", f"{eid}.{key}={value!r}", eid, key)
    return float(value)


def validate_design(context: ProposalContext, design) -> None:
    """Raise ProposalRejected unless `design` only re-geometries the source.

    Checks run before any sorting/indexing by ID, so a duplicate or unhashable
    ID cannot corrupt lookups. Upper bounds/containment stay with the shared
    invariants; this rejects what the legacy plan_valid did not.
    """
    from ..scene import background_cover_allowed
    if not isinstance(design, dict) or not isinstance(design.get("elements"), list) or not isinstance(design.get("canvas"), dict):
        raise ProposalRejected("malformed", "design must be {canvas, elements[]}")
    canvas = design["canvas"]
    for key in ("width", "height"):
        _number(canvas.get(key), "canvas", key)
    if canvas.get("width") != context.target["width"] or canvas.get("height") != context.target["height"]:
        raise ProposalRejected("canvas_mismatch", f"{canvas} != {context.target}")
    originals = {e["id"]: e for e in context.source["elements"]}
    seen = set()
    for el in design["elements"]:
        if not isinstance(el, dict) or not isinstance(el.get("id"), str):
            raise ProposalRejected("malformed", "element without a string id")
        eid = el["id"]
        if eid in seen:
            raise ProposalRejected("duplicate_id", eid, eid)
        seen.add(eid)
        if eid not in originals:
            raise ProposalRejected("unknown_id", eid, eid)
    missing = [i for i in originals if i not in seen]
    if missing:
        raise ProposalRejected("missing_id", ", ".join(sorted(missing)[:6]), missing[0])
    if [e['id'] for e in design['elements']] != [e['id'] for e in context.source['elements']]:
        raise ProposalRejected('element_order','Original paint order must be preserved')
    for el in design["elements"]:
        eid, src = el["id"], originals[el["id"]]
        for key in set(src) | set(el):
            if key in MUTABLE_FIELDS:
                continue
            if src.get(key) == el.get(key):
                continue
            if key == "fit" and background_cover_allowed(context.source, src, el):
                continue
            raise ProposalRejected("non_allowlisted_field", f"{eid}.{key}", eid, key)
        # Same-canvas identity keeps the source's own (possibly bleeding) geometry.
        untouched = context.same_size and all(el.get(k) == src.get(k) for k in ("x", "y", "width", "height"))
        for key in ("x", "y", "width", "height"):
            v = _number(el.get(key), eid, key)
            if key in ("x", "y") and v < -NEGATIVE_TOLERANCE and not untouched:
                raise ProposalRejected("negative_origin", f"{eid}.{key}={v}", eid, key)
            if key in ("width", "height") and v < 0:
                raise ProposalRejected("negative_size", f"{eid}.{key}={v}", eid, key)
        if el.get("type") == "text":
            fs = _number(el.get("font_size"), eid, "font_size")
            if fs < 1:
                raise ProposalRejected("font_size_range", f"{eid}.font_size={fs}", eid, "font_size")


def materialize_patch(context: ProposalContext, patch: dict) -> dict:
    """Apply a geometry patch {element_id: {field: value}} to a source copy.

    The patch is validated field by field (unknown ids, non-allowlisted fields,
    booleans and non-finite numbers are rejected) before any design exists.
    """
    from ..scene import background_cover_allowed
    if not isinstance(patch, dict):
        raise ProposalRejected("malformed", "patch must be a mapping")
    design = {"canvas": dict(context.target), "elements": copy.deepcopy(context.source["elements"])}
    by = {e["id"]: e for e in design["elements"]}
    for eid, fields in patch.items():
        if not isinstance(eid, str) or eid not in by:
            raise ProposalRejected("unknown_id", repr(eid), eid if isinstance(eid, str) else None)
        if not isinstance(fields, dict):
            raise ProposalRejected("malformed", f"{eid} patch is not a mapping", eid)
        for key, value in fields.items():
            if key == "fit" and background_cover_allowed(context.source, by[eid], {**by[eid], "fit": value}):
                by[eid]["fit"] = value
                continue
            if key not in MUTABLE_FIELDS:
                raise ProposalRejected("non_allowlisted_field", f"{eid}.{key}", eid, key)
            v = _number(value, eid, key)
            by[eid][key] = max(1, int(v)) if key == "font_size" else v
    validate_design(context, design)
    return design


# ------------------------------------------------------------------ running
def run_provider(provider, context: ProposalContext, budget=None) -> ProviderResult:
    """Run one provider with failure isolation and per-proposal validation."""
    t0 = time.monotonic()
    name = getattr(provider, "name", type(provider).__name__)
    if budget is not None and getattr(provider, "cost", 0) > 0:
        stop = budget.stop_reason() if hasattr(budget, "stop_reason") else None
        if stop:
            return ProviderResult(name, "skipped", error=stop, elapsed_ms=0.0)
    try:
        # Isolate even an accidentally mutating failed adapter from later
        # providers and the controller's original source/context.
        raw = provider.propose(copy.deepcopy(context), budget)
    except Exception as exc:  # isolation: an optional provider may fail arbitrarily
        return ProviderResult(name, "failed", error=f"{type(exc).__name__}: {exc}",
                              elapsed_ms=(time.monotonic() - t0) * 1000)
    if not isinstance(raw, ProviderResult):
        return ProviderResult(name, "failed", error="provider returned a non-ProviderResult",
                              elapsed_ms=(time.monotonic() - t0) * 1000)
    kept, rejected = [], list(raw.rejections)
    for p in raw.proposals:
        try:
            if not isinstance(p, Proposal):
                raise ProposalRejected("malformed", "not a Proposal record")
            if context.profile.get('background_mode') == 'safe_cover':
                from ..canonical import cover_backdrops
                p = replace(p, design=cover_backdrops(context.source, p.design))
            validate_design(context, p.design)
        except ProposalRejected as exc:
            rejected.append({"template": getattr(p, "template", None), **exc.to_dict()})
            continue
        except Exception as exc:  # validation itself must never crash the run
            rejected.append({"template": getattr(p, "template", None), "code": "malformed", "detail": repr(exc)})
            continue
        kept.append(p)
    status = raw.status
    if status == "ok" and not kept:
        status = "empty"
    return ProviderResult(name, status, tuple(kept), raw.error, (time.monotonic() - t0) * 1000,
                          tuple(rejected), dict(raw.diagnostics))


def run_providers(context: ProposalContext, providers: Iterable, budget=None, *, schedule="cost") -> list[ProviderResult]:
    """Cheap providers first (stable for equal cost); each result is isolated.

    schedule="given" keeps the caller's order (used for exact legacy assembly).
    """
    ordered = list(enumerate(providers))
    if schedule == "cost":
        ordered.sort(key=lambda t: (getattr(t[1], "cost", 0), t[0]))
    return [run_provider(p, context, budget) for _, p in ordered]
