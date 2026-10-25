"""Revision-checked allowlisted edits; arbitrary model documents are not trusted."""
from __future__ import annotations

import copy
import math

from .canonical import MUTABLE_FIELDS
from .contracts import DesignRevision, EditProposal, design_hash
from .invariants import passes, validate
from .scene import background_cover_allowed


def apply_edit(source, revision: DesignRevision, proposal: EditProposal, locks=None):
    if proposal.base_revision != revision.revision_hash or design_hash(revision.design) != revision.revision_hash:
        raise ValueError('Stale or mutated base revision')
    out = copy.deepcopy(revision.design)
    by = {e['id']: e for e in out['elements']}
    originals = {e['id']: e for e in source['elements']}
    if set(proposal.allowed_patch) - set(by): raise ValueError('Unknown edited element')
    for eid, patch in proposal.allowed_patch.items():
        if not isinstance(patch, dict): raise ValueError('Invalid edit patch')
        for key, value in patch.items():
            if key == 'fit' and background_cover_allowed(source, originals[eid], {**by[eid], 'fit': value}):
                by[eid][key] = value
                continue
            if key not in MUTABLE_FIELDS: raise ValueError('Edit changes immutable content or style: ' + key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Edit geometry must be finite numeric values')
            by[eid][key] = max(1, int(value)) if key == 'font_size' else round(float(value), 2)
    for eid, rect in (locks or {}).items():
        if eid not in by or any(abs(by[eid][k] - v) > .5 for k, v in rect.items()):
            raise ValueError('Edit violates exact lock: ' + eid)
    checks = validate(source, out)
    if not passes(checks):
        raise ValueError('; '.join(c['name'] for c in checks if not c['passed'] and c['severity'] == 'error'))
    return out


def proposal_from_design(revision: DesignRevision, design, operation, *, provenance='deterministic'):
    if design['canvas'] != revision.design['canvas']: raise ValueError('Edit changes target canvas')
    old = {e['id']: e for e in revision.design['elements']}
    elements = design['elements']
    if len(elements) != len(old) or {e['id'] for e in elements} != set(old): raise ValueError('Edit changes element coverage')
    if [e['id'] for e in elements] != [e['id'] for e in revision.design['elements']]:
        raise ValueError('Edit changes original paint order')
    patch = {}
    for element in elements:
        before = old[element['id']]
        delta = {k: element.get(k) for k in set(before) | set(element) if before.get(k) != element.get(k)}
        if delta: patch[element['id']] = delta
    return EditProposal(revision.revision_hash, operation, patch, tuple(patch), provenance=provenance)
