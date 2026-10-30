"""Gaurav's engine (vendor/gaurav, commit in vendor/gaurav/COMMIT) as a candidate source.

Both engines are a package called `engine`, so his runs in one helper process (this file as a
script), reused for every resize. `FINAL_GAURAV=0` leaves him out.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / 'vendor/gaurav'
TOP = 6   # his best layouts, at most one per strategy family first

_proc = None


def enabled() -> bool:
    return os.environ.get('FINAL_GAURAV', '1') != '0' and (VENDOR / 'engine').is_dir()


def candidates(source: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    """His top layouts as {'name', 'design'}; empty when he is off or fails."""
    global _proc
    if not enabled():
        return []
    if _proc is None or _proc.poll() is not None:
        _proc = subprocess.Popen([sys.executable, __file__], cwd=VENDOR, text=True, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        atexit.register(_proc.kill)
    try:
        _proc.stdin.write(json.dumps({'source': source, 'target': target}) + '\n')
        _proc.stdin.flush()
        return json.loads(_proc.stdout.readline()).get('candidates', [])
    except (BrokenPipeError, ValueError):
        _proc = None
        return []


def _top(judged) -> list[dict[str, Any]]:
    picked, families = [], set()
    for j in judged:   # best first: the best of each family, then the next best
        if j.fallback or j.strategy in families:
            continue
        families.add(j.strategy)
        picked.append(j)
    for j in judged:
        if len(picked) >= TOP:
            break
        if not j.fallback and j not in picked:
            picked.append(j)
    return [{'name': j.name, 'design': j.design} for j in picked[:TOP]]


def _serve() -> None:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr   # nothing he prints may reach the protocol
    sys.path.insert(0, str(REPO))
    import renderer.converter.convert  # noqa: F401  the one shared renderer
    import renderer.converter.brand_fonts  # noqa: F401
    sys.path.remove(str(REPO))
    sys.path = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).parent]
    sys.path.insert(0, str(VENDOR))
    import engine.backdrop, engine.contrast, engine.render, engine.textfit  # noqa: E401
    from engine.resize import resize_with_candidates
    kit = REPO / 'brand_kit'
    engine.textfit.FONTS_DIR = engine.render.FONTS_DIR = kit / 'fonts'
    engine.contrast.ASSETS_DIR = engine.backdrop.ASSETS_DIR = kit / 'assets'
    for line in sys.stdin:
        request = json.loads(line)
        try:
            _, judged = resize_with_candidates(request['source'], request['target'])
            reply = {'candidates': _top(judged)}
        except Exception as error:  # his engine failing is fewer candidates, not a crash
            reply = {'candidates': [], 'error': repr(error)[:500]}
        real_stdout.write(json.dumps(reply) + '\n')
        real_stdout.flush()


if __name__ == '__main__':
    _serve()
