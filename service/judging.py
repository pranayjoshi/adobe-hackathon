"""Tier 3 AI judge in the viewer (evaluation/judge.py).

Backed by a local OpenJev Multimodal server by default; set
CHAMELEON_VIEWER_JUDGE=ollama to use the Ollama vision model instead.
"""
import os
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from engine.quality import _RENDER_LOCK

router = APIRouter()
BACKEND = os.environ.get('CHAMELEON_VIEWER_JUDGE', 'openjev')


class JudgeRequest(BaseModel):
    source: dict[str, Any]
    design: dict[str, Any]
    samples: int = 1


def _check(judge):
    if BACKEND == 'openjev':
        judge.openjev_check(); return {'model': judge.OPENJEV_MODEL, 'url': judge.OPENJEV_URL}
    judge.ollama_check(); return {'model': judge.OLLAMA_MODEL, 'url': judge.OLLAMA_URL}


@router.get('/api/judge/status')
def status():
    from evaluation import judge
    try:
        return {'available': True, 'backend': BACKEND, **_check(judge)}
    except SystemExit as exc:
        return {'available': False, 'backend': BACKEND, 'detail': str(exc)}


@router.post('/api/judge')
def run(req: JudgeRequest):
    from evaluation import judge
    try:
        _check(judge)
    except SystemExit as exc:
        raise HTTPException(503, str(exc))
    with _RENDER_LOCK:  # the renderer is not thread-safe
        images = [judge.render(req.source), judge.render(req.design)]
    result = judge.score(req.source, req.design, req.design['canvas'], BACKEND, max(1, min(3, req.samples)), images)
    if not result:
        raise HTTPException(502, 'The model returned no valid verdict; try again.')
    return result
