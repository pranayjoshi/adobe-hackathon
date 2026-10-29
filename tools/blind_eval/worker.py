"""Run one engine over a shard of tasks inside its own interpreter.

Usage: python worker.py <engine_name> <engine_root> <renderer_root> <shard.json> <out_dir>

Both engines ship a top-level package called `engine`, so each one runs in a separate
process. Every engine is given the same renderer (`renderer_root`, which has the dataset
font manifest) and the same brand kit, so neither is judged on missing fonts or assets.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

name, engine_root, renderer_root, shard, out_dir = sys.argv[1:6]
engine_root, renderer_root, out_dir = Path(engine_root).resolve(), Path(renderer_root).resolve(), Path(out_dir)
brand_kit = renderer_root / 'brand_kit'

# Import the shared renderer first so the engine's own `renderer` package is never loaded.
sys.path.insert(0, str(renderer_root))
import renderer.converter.convert  # noqa: E402,F401
import renderer.converter.brand_fonts  # noqa: E402,F401
sys.path.remove(str(renderer_root))
sys.path.insert(0, str(engine_root))

from engine.resize import resize  # noqa: E402

if name == 'gaurav':
    # Point the competitor's measuring and judging at the shared fonts and assets.
    import engine.backdrop, engine.contrast, engine.render, engine.textfit  # noqa: E401,E402
    engine.textfit.FONTS_DIR = engine.render.FONTS_DIR = brand_kit / 'fonts'
    engine.contrast.ASSETS_DIR = engine.backdrop.ASSETS_DIR = brand_kit / 'assets'

for task in json.loads(Path(shard).read_text()):
    out = out_dir / f"{task['id']}.json"
    if out.exists():
        continue
    started = time.perf_counter()
    try:
        design = resize(json.loads(json.dumps(task['source'])), task['target_canvas'])
        record = {'ok': True, 'design': design}
    except Exception:
        record = {'ok': False, 'error': traceback.format_exc()[-3000:]}
    record['seconds'] = round(time.perf_counter() - started, 3)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(record))
    tmp.replace(out)
    print(f"{name} {task['id']} {'ok' if record['ok'] else 'FAILED'} {record['seconds']}s", flush=True)
