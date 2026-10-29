"""Chameleon Resize: one design in, every size you need out.

A small, single-purpose front end on the engine. Pick a design, choose sizes, and get the
engine's best layout for each, with a naive-stretch comparison and PNG/JSON/ZIP downloads.

Run from the repo root:
    .venv/bin/uvicorn lite.app:app --port 8770
"""
from __future__ import annotations

import io
import json
import queue
import re
import threading
import time
import uuid
import zipfile
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.canonical import ASSETS_DIR, ROOT
from engine.contracts import NoValidDesign
from engine.pipeline import generate, uniform_stretch
from engine.quality import verify_design
from service.crello import router as crello_router

HERE = ROOT / 'lite'
OUT = ROOT / 'output' / 'lite'
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = [
    {'id': 'ig-post', 'label': 'Instagram Post', 'width': 1080, 'height': 1080},
    {'id': 'ig-story', 'label': 'Story / Reel', 'width': 1080, 'height': 1920},
    {'id': 'yt-thumb', 'label': 'YouTube Thumbnail', 'width': 1280, 'height': 720},
    {'id': 'x-post', 'label': 'X Post', 'width': 1600, 'height': 900},
    {'id': 'pin', 'label': 'Pinterest Pin', 'width': 1000, 'height': 1500},
    {'id': 'li-banner', 'label': 'LinkedIn Banner', 'width': 1584, 'height': 396},
    {'id': 'web-hero', 'label': 'Website Hero', 'width': 1600, 'height': 400},
    {'id': 'billboard', 'label': 'Billboard Ad', 'width': 970, 'height': 250},
    {'id': 'mrec', 'label': 'Medium Rectangle', 'width': 300, 'height': 250},
    {'id': 'half-page', 'label': 'Half Page Ad', 'width': 300, 'height': 600},
]

app = FastAPI(title='Chameleon Resize', version='1.0')
app.include_router(crello_router)


class Size(BaseModel):
    width: int = Field(ge=16, le=4096)
    height: int = Field(ge=16, le=4096)
    label: str = Field(default='', max_length=80)


class ResizeRequest(BaseModel):
    source: dict[str, Any]
    sizes: list[Size] = Field(min_length=1, max_length=12)


class PreviewRequest(BaseModel):
    design: dict[str, Any]


JOBS: dict[str, dict[str, Any]] = {}
_queue: queue.Queue = queue.Queue()


def _check(design):
    """Reject anything the engine cannot read; the engine itself judges the rest."""
    ok = isinstance(design, dict) and isinstance(design.get('canvas'), dict) and isinstance(design.get('elements'), list)
    if not ok or not 1 <= len(design['elements']) <= 200:
        raise HTTPException(422, 'Not a Chameleon design: it needs a "canvas" and 1-200 "elements".')
    if not all(isinstance(design['canvas'].get(k), (int, float)) and 1 <= design['canvas'][k] <= 8000 for k in ('width', 'height')):
        raise HTTPException(422, 'Canvas width and height must be between 1 and 8000.')
    required = {'text': ('content', 'font_family', 'font_size', 'color'), 'image': ('asset_id',), 'shape': ('shape_type', 'fill_color')}
    ids = set()
    for e in design['elements']:
        if not isinstance(e, dict) or not isinstance(e.get('id'), str) or e['id'] in ids:
            raise HTTPException(422, 'Every element needs a unique string "id".')
        ids.add(e['id'])
        if e.get('type') not in required or any(k not in e for k in required[e['type']]):
            raise HTTPException(422, f"Element '{e['id']}' is missing required fields.")
        if any(not isinstance(e.get(k), (int, float)) for k in ('x', 'y', 'width', 'height')) or e['width'] <= 0 or e['height'] <= 0:
            raise HTTPException(422, f"Element '{e['id']}' needs numeric x, y, width and height.")
        if e['type'] == 'image' and not (ASSETS_DIR / str(e['asset_id'])).resolve().is_relative_to(ASSETS_DIR.resolve()):
            raise HTTPException(422, 'Image assets must come from brand_kit/assets.')


def _save_render(design, folder, name):
    """Render with the official renderer; returns the PNG file name, or None."""
    result = verify_design(design)
    image = result.get('image')
    if not result.get('verified') or image is None:
        return None
    image.save(folder / f'{name}.png', format='PNG')
    return f'{name}.png'


def _run(rid):
    job = JOBS[rid]
    folder = OUT / rid
    folder.mkdir(parents=True, exist_ok=True)
    source = job['source']
    (folder / 'source.json').write_text(json.dumps(source))
    job['state'] = 'running'
    for i, item in enumerate(job['items']):
        item['state'] = 'running'
        target = {'width': item['width'], 'height': item['height']}
        t0 = time.monotonic()
        try:
            stretch = uniform_stretch(source, target)
            item['stretch'] = _save_render(stretch, folder, f'{i}-stretch')
            try:
                cands = generate(source, target, top_k=1)
            except NoValidDesign:
                cands = []
            best = next((c for c in cands if c.get('valid') and c.get('rubric', {}).get('gate', True)), None)
            if best is None:
                item.update(state='failed', error='No layout passed every check for this size.')
                continue
            (folder / f'{i}.json').write_text(json.dumps(best['design'], indent=2))
            png = _save_render(best['design'], folder, str(i))
            if png is None:
                item.update(state='failed', error='The renderer could not draw this layout.')
                continue
            item.update(state='done', render=png, design=f'{i}.json', method=best.get('template', ''))
        except Exception as exc:  # one bad size must not end the batch
            item.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            item['seconds'] = round(time.monotonic() - t0, 1)
    job['state'] = 'done'


def _worker():
    while True:
        _run(_queue.get())


threading.Thread(target=_worker, daemon=True).start()


def _safe(rid):
    if not re.fullmatch(r'[a-f0-9]{12}', rid) or rid not in JOBS:
        raise HTTPException(404, 'Unknown job')
    return JOBS[rid]


@app.get('/api/presets')
def presets():
    return PRESETS


@app.get('/api/samples')
def samples():
    out = []
    for p in sorted((ROOT / 'input').glob('*.json')):
        t = json.loads(p.read_text())
        out.append({'id': p.stem, 'title': t.get('difficulty_label') or p.stem, 'source': t['source']})
    return out


@app.post('/api/preview')
def preview(req: PreviewRequest):
    _check(req.design)
    folder = OUT / 'previews'
    folder.mkdir(exist_ok=True)
    name = uuid.uuid4().hex[:12]
    png = _save_render(req.design, folder, name)
    if png is None:
        raise HTTPException(422, 'The renderer could not draw this design (missing image or font?).')
    return {'render': f'/files/previews/{png}'}


@app.post('/api/resize')
def resize(req: ResizeRequest):
    _check(req.source)
    rid = uuid.uuid4().hex[:12]
    JOBS[rid] = {'id': rid, 'state': 'queued', 'source': req.source, 'created': time.time(),
                 'items': [{'width': s.width, 'height': s.height, 'label': s.label or f'{s.width}×{s.height}',
                            'state': 'queued'} for s in req.sizes]}
    _queue.put(rid)
    return {'id': rid}


@app.get('/api/jobs/{rid}')
def job(rid: str):
    j = _safe(rid)
    base = f'/files/{rid}/'
    items = [{**it, **{k: base + it[k] for k in ('render', 'stretch', 'design') if it.get(k)}} for it in j['items']]
    return {'id': rid, 'state': j['state'], 'items': items}


@app.get('/api/jobs/{rid}/zip')
def job_zip(rid: str):
    j = _safe(rid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, it in enumerate(j['items']):
            stem = f"{it['width']}x{it['height']}"
            for key, ext in (('render', 'png'), ('design', 'json')):
                if it.get(key):
                    z.write(OUT / rid / it[key], f'{stem}.{ext}')
    return Response(buf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="chameleon-{rid}.zip"'})


app.mount('/files', StaticFiles(directory=str(OUT)), name='files')


@app.get('/')
def index():
    return FileResponse(HERE / 'index.html')
