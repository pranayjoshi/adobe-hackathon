"""Gaurav's engine as one more candidate generator for the merged engine.

Both engines ship a top-level package called `engine`, so his runs in a helper process
(this file, run as a script) that answers one JSON request per line on stdin. The helper
starts once per merged process and is reused for every resize.

`MERGED_GAURAV` is the path of his `chameleon/` checkout (default: the sibling clone at
../chameleon-competitors/gaurav/chameleon); `MERGED_GAURAV=0` turns the bridge off.
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
DEFAULT = REPO.parent / 'chameleon-competitors/gaurav/chameleon'


def root() -> Path | None:
    value = os.environ.get('MERGED_GAURAV', str(DEFAULT))
    if value == '0' or not Path(value).is_dir():
        return None
    return Path(value).resolve()


_proc = None


def resize(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any] | None:
    """Gaurav's layout for this size, or None when his engine is off, missing or fails."""
    global _proc
    where = root()
    if where is None:
        return None
    if _proc is None or _proc.poll() is not None:
        _proc = subprocess.Popen([sys.executable, __file__, str(where), str(REPO)], cwd=where, text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        atexit.register(_proc.kill)
    try:
        _proc.stdin.write(json.dumps({'source': source, 'target': target}) + '\n')
        _proc.stdin.flush()
        reply = json.loads(_proc.stdout.readline())
    except (BrokenPipeError, ValueError):
        _proc = None
        return None
    return reply.get('design')


def _serve(where: Path, repo: Path) -> None:
    # Measure and judge with this repository's brand kit, as our engine does.
    sys.path.insert(0, str(where))
    import engine.backdrop, engine.contrast, engine.render, engine.textfit  # noqa: E401
    from engine.resize import resize as theirs
    kit = repo / 'brand_kit'
    if kit.is_dir():
        engine.textfit.FONTS_DIR = engine.render.FONTS_DIR = kit / 'fonts'
        engine.contrast.ASSETS_DIR = engine.backdrop.ASSETS_DIR = kit / 'assets'
    for line in sys.stdin:
        request = json.loads(line)
        try:
            reply = {'design': theirs(request['source'], request['target'])}
        except Exception as error:  # his engine failing is a missing candidate, not a crash
            reply = {'error': repr(error)[:500]}
        sys.stdout.write(json.dumps(reply) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    _serve(Path(sys.argv[1]), Path(sys.argv[2]))
