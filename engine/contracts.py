"""Small versioned contracts for bounded, observable design editing.

Design documents stay schema-pure; all orchestration state lives in sidecars.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

SCHEMA_VERSION = 1
CONTROLLER_MODES = ('legacy_batch', 'single', 'single_restart', 'small_beam')


def design_hash(design: dict) -> str:
    document = {'canvas': design['canvas'], 'elements': design['elements']}
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class NoValidDesign(ValueError):
    """No fully verified structurally valid output; never an export success."""

    def __init__(self, message='No verified valid design', *, report=None):
        super().__init__(message)
        self.report = report or {}


@dataclass(frozen=True)
class EditProposal:
    base_revision: str
    operation: str
    allowed_patch: dict[str, dict[str, float]]
    affected_ids: tuple[str, ...] = ()
    intended_improvement: str = ''
    provenance: str = 'deterministic'


@dataclass(frozen=True)
class DesignRevision:
    revision_hash: str
    parent_revision: str | None
    design: dict
    origin: str
    verification_status: str = 'unverified'

    @classmethod
    def create(cls, design, parent=None, origin='deterministic'):
        snapshot = copy.deepcopy({'canvas': design['canvas'], 'elements': design['elements']})
        return cls(design_hash(snapshot), parent, snapshot, origin)


@dataclass
class Budget:
    seconds: float = 12.0
    max_edits: int = 24
    max_renders: int = 8
    max_model_calls: int = 4
    cancellation: Callable[[], bool] | None = field(default=None, repr=False)
    started: float = field(default_factory=time.monotonic)
    edits: int = 0
    renders: int = 0
    model_calls: int = 0

    def __post_init__(self):
        if isinstance(self.seconds, bool) or not math.isfinite(self.seconds) or not 0 < self.seconds <= 300:
            raise ValueError('controller_seconds must be finite and in (0, 300]')
        for name, maximum in [('max_edits', 500), ('max_renders', 100), ('max_model_calls', 100)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [0, {maximum}]')

    def stop_reason(self):
        if self.cancellation and self.cancellation(): return 'cancelled'
        if time.monotonic() - self.started >= self.seconds: return 'deadline'
        if self.edits >= self.max_edits: return 'edit_budget'
        if self.renders >= self.max_renders: return 'render_budget'
        return None

    def remaining(self):
        return max(0.0, self.seconds - (time.monotonic() - self.started))

    def usage(self):
        return {'edits': self.edits, 'renders': self.renders, 'model_calls': self.model_calls,
                'seconds': round(time.monotonic() - self.started, 3),
                'limits': {'seconds': self.seconds, 'edits': self.max_edits,
                           'renders': self.max_renders, 'model_calls': self.max_model_calls}}


@dataclass(frozen=True)
class PairVerdict:
    before_hash: str
    after_hash: str
    verdict: str
    evaluator_version: str
    provenance: str = 'model'
    available: bool = True
    details: dict = field(default_factory=dict)


class SessionRepository(Protocol):
    def append_event(self, session_id: str, event: dict, expected_revision: str | None = None) -> None: ...


def json_record(record: Any) -> dict:
    return asdict(record)
