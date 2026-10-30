"""Hybrid entry point: `resize(source, target_canvas) -> design`, best of both engines.

Our engine runs in this process. Gaurav's runs at the same time in its own interpreter,
because both ship a top-level package named `engine`; set CHAMELEON_GAURAV_ROOT to its
`chameleon/` folder. Each layout is repaired and the cleaner one is kept (hybrid.combine).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hybrid.combine import choose

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'tools/blind_eval/worker.py'
GAURAV_ROOT = Path(os.environ.get('CHAMELEON_GAURAV_ROOT') or ROOT.parent / 'chameleon-competitors/gaurav/chameleon')


def _ours(source, target):
    from engine.contracts import NoValidDesign
    from engine.resize import resize as ours
    try:
        return ours(source, target)
    except NoValidDesign:
        return None


def _gaurav(source, target, timeout=120):
    if not GAURAV_ROOT.is_dir():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / 'shard.json'  # the worker writes its result to task.json
        shard.write_text(json.dumps([{'id': 'task', 'source': source, 'target_canvas': target}]))
        subprocess.run([sys.executable, str(WORKER), 'gaurav', str(GAURAV_ROOT), str(ROOT), str(shard), tmp],
                       cwd=GAURAV_ROOT, capture_output=True, timeout=timeout)
        out = Path(tmp) / 'task.json'
        rec = json.loads(out.read_text()) if out.is_file() else {'ok': False}
        return rec['design'] if rec.get('ok') else None


def resize_with_report(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    target = {'width': int(target_canvas['width']), 'height': int(target_canvas['height'])}
    with ThreadPoolExecutor(2) as pool:
        theirs = pool.submit(_gaurav, source, target)
        candidates = {'pranay': _ours(source, target), 'gaurav': theirs.result()}
    return choose(source, candidates, ROOT / 'brand_kit/assets')


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    return resize_with_report(source, target_canvas)['design']
