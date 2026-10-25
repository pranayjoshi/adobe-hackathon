"""Adapters wrapping the legacy proposal sources behind the Provider contract.

Each adapter reproduces exactly what engine/pipeline.py:generate assembles
today for its slice of the portfolio, so `legacy_proposals` can replace that
inline block with no behavior change. Costs order initializer scheduling:
cheap geometric baselines first, model calls last.
"""
from __future__ import annotations

import copy
import time

from .base import Proposal, ProposalContext, ProviderResult


def _result(name, proposals, t0, **kw) -> ProviderResult:
    return ProviderResult(name, kw.pop("status", "ok" if proposals else "empty"), tuple(proposals),
                          elapsed_ms=(time.monotonic() - t0) * 1000, **kw)


class IdentityProvider:
    """Same-canvas tasks keep the original composition (prevents needless redesign)."""
    name, cost = "identity", 0

    def propose(self, context: ProposalContext, budget=None) -> ProviderResult:
        t0 = time.monotonic()
        if not context.same_size:
            return _result(self.name, [], t0, status="skipped", error="target differs from source canvas")
        design = copy.deepcopy(context.source)
        design["canvas"] = dict(context.target)  # pipeline.build applies the same envelope
        return _result(self.name, [Proposal(self.name, "identity", "Original composition",
                                            "Already at the requested size", design, "identity")], t0)


def source_frame_design(source, target, A) -> dict:
    """Uniform scale of the full original frame (verbatim pipeline logic)."""
    from ..canonical import element_box, rotated_extent, union_all
    safe = copy.deepcopy(source); safe['canvas'] = dict(target)
    frame = union_all([rotated_extent(e) for e in source['elements']] + [element_box({'x': 0, 'y': 0, 'width': A.sw, 'height': A.sh})])
    k = min(target['width'] / frame.w, target['height'] / frame.h)
    ox = (target['width'] - frame.w * k) / 2 - frame.x * k; oy = (target['height'] - frame.h * k) / 2 - frame.y * k
    for e in safe['elements']:
        e.update(x=e['x'] * k + ox, y=e['y'] * k + oy, width=e['width'] * k, height=e['height'] * k)
        if e['type'] == 'text': e['font_size'] = max(1, int(e['font_size'] * k))
    return safe


class SourceFrameProvider:
    """Original arrangement, uniformly scaled; a real safety candidate, gated like the rest."""
    name, cost = "source_frame", 4

    def propose(self, context, budget=None):
        t0 = time.monotonic()
        d = source_frame_design(context.source, context.target, context.analysis)
        return _result(self.name, [Proposal(self.name, "source_frame", "Original arrangement",
                                            "Uniform scaling preserves the original arrangement; may leave unused space.",
                                            d, "source_frame")], t0)


class TemplateProvider:
    """engine.layout.propose structural templates, serialised exactly like pipeline."""
    name, cost = "templates", 1

    def __init__(self, portfolio: str | None = None):
        self.portfolio = portfolio  # None -> CHAMELEON_PROPOSAL_PORTFOLIO (default legacy)

    def propose(self, context, budget=None):
        from ..layout import propose
        from ..pipeline import serialise
        t0 = time.monotonic()
        A, tgt = context.analysis, context.target
        out, report = [], {}
        for P in propose(A, tgt["width"], tgt["height"], context.emphasis, portfolio=self.portfolio, report=report):
            fam = P.template.split("_scale")[0]
            out.append(Proposal(self.name, P.template, P.name, P.description, serialise(context.source, tgt, P), fam,
                                provenance={"relaxed_floors": P.relaxed_floors}))
        return _result(self.name, out, t0, diagnostics=report)


class RetrievalProvider:
    """engine.retrieval.retrieve; an invalid library is a typed failure, never a crash."""
    name, cost = "retrieval", 3

    def __init__(self, version: str | None = None):
        self.version = version  # None -> CHAMELEON_RETRIEVAL_VERSION (default legacy)

    def propose(self, context, budget=None):
        from ..retrieval import retrieve_with_report
        t0 = time.monotonic()
        if not context.profile.get("retrieval", True):
            return _result(self.name, [], t0, status="skipped", error="retrieval disabled")
        try:
            designs, report = retrieve_with_report(context.source, context.target, context.analysis, version=self.version)
        except (ValueError, OSError, KeyError) as exc:
            # pipeline records {"retrieval": "invalid library ignored"} and keeps templates.
            return _result(self.name, [], t0, status="failed", error=f"invalid library ignored: {exc}")
        selected = report.get("selected") or []
        out = [Proposal(self.name, f"retrieved_{i}", "Retrieved arrangement", "Matched to training geometry", d,
                        "retrieved", provenance=selected[i] if i < len(selected) else {})
               for i, d in enumerate(designs)]
        return _result(self.name, out, t0, diagnostics=report)


class PhotoSceneProvider:
    """Read-only call into engine.scene.propose_photo_scene (owned by lane C3)."""
    name, cost = "photo_scene", 2

    def propose(self, context, budget=None):
        from ..scene import propose_photo_scene
        t0 = time.monotonic();report={}
        out = [Proposal(self.name, 'photo_scene', 'Photo with copy column',
                        'Original background cropped without distortion; copy stays in its original horizontal region.', d,
                        "photo_scene")
               for d in propose_photo_scene(context.source, context.target, context.analysis,
                                            mode=context.profile.get('photo_mode'),budget=budget,status=report)]
        return _result(self.name, out, t0, diagnostics=report)


class VLMProvider:
    """Optional local visual model; off unless the profile enables it."""
    name, cost = "vlm", 5

    def propose(self, context, budget=None):
        from ..neural_proposals import propose as neural
        t0 = time.monotonic()
        if not context.profile.get("vlm", False):
            return _result(self.name, [], t0, status="skipped", error="vlm disabled")
        if budget is not None:
            if budget.model_calls >= budget.max_model_calls or budget.renders+2 > budget.max_renders or budget.remaining() <= .05:
                return _result(self.name, [], t0, status="skipped", error="model_budget")
            budget.model_calls += 1
            budget.renders += 1
        d, event = neural(context.source, context.target, context.analysis, context.locks,
                          timeout=min(90,budget.remaining()) if budget is not None else 90,
                          cancelled=budget.cancellation if budget is not None else None)
        if budget is not None:
            budget.model_calls += int(event.get('model_calls',1)) - 1
            budget.renders += int(event.get('renders',1)) - 1
        if not d:
            return _result(self.name, [], t0, status="empty", error=str((event or {}).get("status")), diagnostics={"vlm": event})
        return _result(self.name, [Proposal(self.name, "visual_proposal", "Visual proposal",
                                            "Suggested by the configured local model", d, "visual_proposal")], t0,
                       diagnostics={"vlm": event})


def default_providers(portfolio=None, retrieval_version=None):
    """Legacy portfolio order is pipeline order; costs drive initializer order."""
    return [TemplateProvider(portfolio), RetrievalProvider(retrieval_version), VLMProvider(),
            PhotoSceneProvider(), IdentityProvider(), SourceFrameProvider()]
