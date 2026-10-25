"""Bounded edit/evaluate controller with independently verified checkpoints.

This module schedules injected providers, assessments and verifiers. It never
trains on its own decisions and never turns a failed render into a success.
"""
from __future__ import annotations

import copy
import uuid
from typing import Callable, Iterable

from .contracts import Budget, CONTROLLER_MODES, DesignRevision, NoValidDesign, design_hash
from .materialize import apply_edit, proposal_from_design


def candidate_key(candidate):
    from .refine import key
    return key(candidate)


def better(after, before):
    """Never compare a learned logit with a rubric score on another scale."""
    from .learning import preference_comparison
    a, b = candidate_key(after), candidate_key(before)
    if a[:3] != b[:3]: return a[:3] > b[:3]
    preference = preference_comparison(before, after)
    if preference.get('available') and preference.get('verdict') in ('before', 'after'):
        return preference['verdict'] == 'after'
    return a[3:] > b[3:]


def run_controller(source, seeds: Iterable[tuple[str, dict]], *, assess: Callable,
                   expand: Callable, verify: Callable, mode='single_restart',
                   locks=None, budget=None, target_score=95.0, compare=None,
                   progress_callback=None, artifact_callback=None, repository=None):
    if mode not in CONTROLLER_MODES[1:]: raise ValueError('Unknown controller mode')
    budget = budget or Budget()
    locks = locks or {}
    session = uuid.uuid4().hex
    events, seen, beam = [], set(), []
    incumbent = None
    active = None
    stop = None
    seed_iter = iter(seeds)

    def emit(kind, **payload):
        event = {'sequence': len(events), 'type': kind, **payload}
        events.append(copy.deepcopy(event))
        if repository: repository.append_event(session, event)
        if progress_callback: progress_callback(copy.deepcopy(event))

    def report():
        return {'schema_version': 1, 'session_id': session, 'mode': mode, 'stop_reason': stop,
                'active_revision': design_hash(active['design']) if active else None,
                'incumbent_revision': design_hash(incumbent['design']) if incumbent else None,
                'events': events, 'usage': budget.usage()}

    def evaluate(design, operation, parent=None, is_edit=False):
        nonlocal incumbent
        try:
            hashed = design_hash(design)
            if hashed in seen:
                emit('rejected', operation=operation, reason='duplicate', revision_hash=hashed)
                return None
            seen.add(hashed)
            candidate = assess(design, operation)
        except (ValueError, KeyError, TypeError) as exc:
            emit('rejected', operation=operation, reason='invalid', detail=str(exc))
            return None
        valid = bool(candidate['valid'] and candidate['rubric']['gate'])
        promising = valid and (incumbent is None or better(candidate, incumbent))
        verification = {'verified': False, 'status': 'not_rendered'}
        image = None
        if promising and budget.renders < budget.max_renders and budget.remaining() > 0:
            if not (budget.cancellation and budget.cancellation()):
                budget.renders += 1
                verification = verify(candidate['design'])
                image = verification.pop('image', None)
                candidate['verification'] = verification
        valid = bool(candidate['valid'] and candidate['rubric']['gate'])
        promising = valid and (incumbent is None or better(candidate, incumbent))
        accepted = promising and bool(verification.get('verified'))
        # A fallible visual judge may veto a soft preference improvement, but
        # cannot approve a mechanical regression or override preservation.
        if accepted and compare and incumbent is not None:
            if budget.model_calls < budget.max_model_calls and budget.remaining() > 0:
                budget.model_calls += 1
                try:
                    verdict = compare(incumbent['design'], candidate['design'])
                except Exception as exc:
                    verdict = {'available': False, 'reason': type(exc).__name__}
                actual_calls = verdict.get('provenance', {}).get('model_calls', 1)
                if isinstance(actual_calls, int) and actual_calls >= 0:
                    budget.model_calls += actual_calls - 1
                emit('comparison', revision_hash=hashed, verdict=verdict)
                if verdict.get('available') and verdict.get('verdict') in ('A', 'a', 'before', 'both_bad'):
                    if candidate_key(candidate)[:3] == candidate_key(incumbent)[:3]: accepted = False
        emit('accepted' if accepted else 'assessed', revision_hash=hashed, parent_revision=parent,
             operation=operation, valid=valid, verified=bool(verification.get('verified')),
             key=list(candidate_key(candidate)), verification=verification)
        if artifact_callback:
            artifact_callback({'type': 'revision', 'revision_hash': hashed, 'parent_revision': parent,
                               'operation': operation, 'design': copy.deepcopy(candidate['design']),
                               'accepted': accepted, 'verified': bool(verification.get('verified')),
                               'rendered': image is not None, 'image': image,
                               'assessment': {'valid': valid, 'rubric': candidate['rubric'],
                                              'score': candidate['score'], 'verification': verification}})
        if accepted:
            incumbent = candidate
        return candidate if valid else None

    def next_seed():
        # Recovery is bounded even if every initializer is invalid. One
        # trajectory may recover until it has its first valid starting state.
        for operation, design in seed_iter:
            if budget.stop_reason(): return None
            budget.edits += 1
            result = evaluate(design, operation)
            if result is not None: return result
        return None

    emit('stage', stage='initializing')
    active = next_seed()
    if active: beam = [active]
    while active is not None:
        stop = budget.stop_reason()
        if stop: break
        if incumbent:
            k = candidate_key(incumbent)
            score = (incumbent['rubric'].get('tier2') or {}).get('overall', 0)
            if k[1:3] == (0, 0) and score >= target_score:
                stop = 'threshold'; break
        improved = False
        expanded = []
        emit('stage', stage='editing', revision_hash=design_hash(active['design']))
        for base in beam:
            revision = DesignRevision.create(base['design'])
            for operation, proposed in expand(base, budget):
                if budget.stop_reason(): break
                budget.edits += 1
                try:
                    proposal = proposal_from_design(revision, proposed, operation)
                    materialized = apply_edit(source, revision, proposal, locks)
                except (ValueError, KeyError, TypeError) as exc:
                    emit('rejected', operation=operation, reason='invalid_edit', detail=str(exc))
                    continue
                before = design_hash(incumbent['design']) if incumbent else None
                result = evaluate(materialized, operation, revision.revision_hash, True)
                if result:
                    expanded.append(result)
                    if incumbent and design_hash(incumbent['design']) != before:
                        improved = True
                        if mode != 'small_beam': break
            if improved and mode != 'small_beam': break
        if improved:
            active = incumbent
            beam = [incumbent]
            if mode == 'small_beam':
                for candidate in sorted(expanded, key=candidate_key, reverse=True):
                    if candidate is not incumbent:
                        beam.append(candidate)
                    if len(beam) >= 3: break
            continue
        if mode == 'small_beam' and expanded:
            beam = sorted(expanded, key=candidate_key, reverse=True)[:3]
            active = beam[0]
            continue
        if mode in ('single_restart', 'small_beam') and not budget.stop_reason():
            emit('stage', stage='restarting')
            active = next_seed()
            beam = [active] if active else []
            if active: continue
        stop = budget.stop_reason() or 'stagnation'
        break
    stop = stop or budget.stop_reason() or 'no_valid_design'
    emit('stopped', stop_reason=stop, usage=budget.usage())
    result_report = report()
    if incumbent is None:
        raise NoValidDesign('No verified valid design: ' + stop, report=result_report)
    incumbent = copy.deepcopy(incumbent)
    incumbent['controller'] = result_report
    incumbent['trace'] = incumbent.get('trace', []) + [{'controller': result_report}]
    return [incumbent]
