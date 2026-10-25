"""Proposal providers (plan 03/05): typed contract, legacy adapters, diversity.

Nothing here imports the controller; the controller consumes `Initializer`.
"""
from .base import (CONTRACT_VERSION, Proposal, ProposalContext, ProposalRejected, Provider, ProviderResult,
                   build_context, materialize_patch, run_provider, run_providers, validate_design)
from .adapters import (IdentityProvider, PhotoSceneProvider, RetrievalProvider, SourceFrameProvider,
                       TemplateProvider, VLMProvider, default_providers, source_frame_design)
from .diversity import FINGERPRINT_VERSION, Fingerprint, dedupe, distance, fingerprint, mmr_select
from .portfolio import Initializer, legacy_proposals, seeds

__all__ = [n for n in dir() if not n.startswith("_")]
