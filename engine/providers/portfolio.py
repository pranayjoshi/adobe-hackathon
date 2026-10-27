"""Portfolio assembly: the exact legacy proposal list, and controller initializers.

`legacy_proposals(context)` returns the same (template, name, description,
design) list, in the same order, that engine/pipeline.py:generate builds
inline today -- a drop-in replacement for that block. `Initializer` yields
`(operation, design)` seeds for engine.controller.run_controller(seeds=...),
cheap providers first, with near-duplicate geometry skipped so restarts move
to genuinely different compositions.
"""
from __future__ import annotations

import copy
import time
from typing import Iterator

from .adapters import (IdentityProvider, PhotoSceneProvider, RetrievalProvider, SourceFrameProvider,
                       TemplateProvider, VLMProvider, default_providers)
from .base import ProposalContext, ProviderResult, build_context, run_providers
from .diversity import NEAR_DUPLICATE, distance, fingerprint


def legacy_proposals(context: ProposalContext, *, portfolio=None, retrieval_version=None):
    """(proposals, events, results) matching pipeline.generate's assembly.

    proposals: list of (template, name, description, design)
    events:    the pipeline `events` entries (retrieval / vlm)
    results:   per-provider ProviderResult records (status, latency, rejections)
    """
    templates, retrieval, vlm, photo, identity, frame = default_providers(portfolio, retrieval_version)
    results = run_providers(context, [templates, retrieval, vlm, photo, identity, frame], schedule="given")
    by = {r.provider: r for r in results}
    proposals, events = [], []
    proposals += [p.as_legacy() for p in by["templates"].proposals]
    r = by["retrieval"]
    if r.status == "failed":
        events.append({"retrieval": "invalid library ignored"})
    proposals += [p.as_legacy() for p in r.proposals]
    if context.profile.get("vlm", False):
        events.append({"vlm": by["vlm"].diagnostics.get("vlm", {"status": by["vlm"].error})})
        proposals += [p.as_legacy() for p in by["vlm"].proposals]
    proposals += [p.as_legacy() for p in by["photo_scene"].proposals]
    if by["identity"].proposals:
        proposals.insert(0, by["identity"].proposals[0].as_legacy())
    proposals += [p.as_legacy() for p in by["source_frame"].proposals]
    from ..retarget.propose import candidates as learned
    proposals += learned(context.source, context.target, context.analysis, context.profile, events)
    return proposals, events, results


class Initializer:
    """Controller seed source: `Initializer(...)(context, budget)` -> Iterator[(operation, design)].

    Matches the `seeds` iterable engine.controller.run_controller consumes.
    Providers run lazily in cost order, so an exhausted budget never pays for
    expensive providers. Template/retrieval seeds are repaired exactly as the
    pipeline repairs them (identity/source_frame are evaluated raw there too).
    Provider outcomes are appended to `self.results` for tracing.
    """

    RAW = ("identity", "source_frame")

    def __init__(self, providers=None, *, portfolio=None, retrieval_version=None, repair=True,
                 near_duplicate=NEAR_DUPLICATE):
        self.providers = list(providers) if providers is not None else default_providers(portfolio, retrieval_version)
        self.repair, self.near_duplicate = repair, near_duplicate
        self.results: list[ProviderResult] = []

    def __call__(self, context: ProposalContext, budget=None) -> Iterator[tuple[str, dict]]:
        return self.propose(context, budget)

    def propose(self, context: ProposalContext, budget=None) -> Iterator[tuple[str, dict]]:
        from ..pipeline import _lock
        from ..repair import repair as repair_design
        prints = []
        ordered = sorted(enumerate(self.providers), key=lambda t: (getattr(t[1], "cost", 0), t[0]))
        for _, provider in ordered:
            [result] = run_providers(context, [provider], budget)
            self.results.append(result)
            for p in result.proposals:
                design = copy.deepcopy(p.design)
                if self.repair and p.template not in self.RAW:
                    design, _ = repair_design(context.source, _lock(design, context.analysis, context.locks),
                                              context.analysis, context.locks)
                design = {"canvas": dict(context.target), "elements": design["elements"]}
                fp = fingerprint(design, context.analysis, p.configuration_ids)
                if any(distance(fp, q) < self.near_duplicate for q in prints):
                    continue
                prints.append(fp)
                yield f"initialize/{result.provider}/{p.template}", design


def seeds(source, target, analysis=None, *, locks=None, options=None, budget=None, **kw):
    """Convenience: build the context and return the seed iterator for run_controller."""
    context = build_context(source, target, analysis, locks=locks, options=options)
    return Initializer(**kw)(context, budget)
