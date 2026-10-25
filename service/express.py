"""Declared editable-export profile; actual SDK preflight stays in the sandbox.

Both the endpoint and browser adapter use the same checked-in capability JSON.
This cannot establish host font availability or rendering parity.
"""
import json
from pathlib import Path
from fastapi import APIRouter
router=APIRouter()
PROFILE=Path(__file__).resolve().parents[1]/'express_addon'/'src'/'capabilities.json'

@router.get('/api/express/capabilities')
def capabilities():
    return json.loads(PROFILE.read_text())
