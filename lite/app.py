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
from engine.quality import unverified_missing_asset_preview, verify_design
from service.crello import router as crello_router

HERE = ROOT / 'lite'
OUT = ROOT / 'output' / 'lite'
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = [  # the same catalogue as the main app
    {'id': 'ig-post', 'group': 'Social', 'label': 'Instagram Post', 'width': 1080, 'height': 1080},
    {'id': 'ig-portrait', 'group': 'Social', 'label': 'Instagram Portrait', 'width': 1080, 'height': 1350},
    {'id': 'ig-story', 'group': 'Social', 'label': 'Story / Reel', 'width': 1080, 'height': 1920},
    {'id': 'yt-thumb', 'group': 'Social', 'label': 'YouTube Thumbnail', 'width': 1280, 'height': 720},
    {'id': 'x-post', 'group': 'Social', 'label': 'X / Twitter Post', 'width': 1600, 'height': 900},
    {'id': 'li-banner', 'group': 'Social', 'label': 'LinkedIn Banner', 'width': 1584, 'height': 396},
    {'id': 'fb-cover', 'group': 'Social', 'label': 'Facebook Cover', 'width': 1640, 'height': 624},
    {'id': 'pin', 'group': 'Social', 'label': 'Pinterest Pin', 'width': 1000, 'height': 1500},
    {'id': 'web-hero', 'group': 'Web', 'label': 'Website Hero', 'width': 1600, 'height': 400},
    {'id': 'leaderboard', 'group': 'Display ads', 'label': 'Leaderboard', 'width': 728, 'height': 90},
    {'id': 'billboard', 'group': 'Display ads', 'label': 'Billboard', 'width': 970, 'height': 250},
    {'id': 'mrec', 'group': 'Display ads', 'label': 'Medium Rectangle', 'width': 300, 'height': 250},
    {'id': 'skyscraper', 'group': 'Display ads', 'label': 'Wide Skyscraper', 'width': 160, 'height': 600},
    {'id': 'half-page', 'group': 'Display ads', 'label': 'Half Page', 'width': 300, 'height': 600},
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


# Targets whose aspect ratio is within this fraction of the source's count as the same shape.
SAME_SHAPE = 0.03


def same_shape(source, target):
    """True when the target has the source's proportions, give or take SAME_SHAPE."""
    a = source['canvas']['width'] / source['canvas']['height']
    b = target['width'] / target['height']
    return abs(b / a - 1) <= SAME_SHAPE


def scale_design(source, target):
    """The source, scaled to a same-shape target: the identical design at a new size.

    Geometry follows each axis so layers still meet the canvas edges; text sizes, letter
    spacing, strokes and corner radii follow the smaller scale so nothing grows past its box.
    A target of the source's exact size returns the source unchanged.
    """
    W, H = source['canvas']['width'], source['canvas']['height']
    if (target['width'], target['height']) == (W, H):
        return json.loads(json.dumps(source))
    sx, sy = target['width'] / W, target['height'] / H
    s = min(sx, sy)
    out = json.loads(json.dumps(source))
    out['canvas'] = {**out['canvas'], 'width': target['width'], 'height': target['height']}
    for e in out['elements']:
        e.update(x=e['x'] * sx, y=e['y'] * sy, width=e['width'] * sx, height=e['height'] * sy)
        if e['type'] == 'text':
            e['font_size'] = max(1.0, e['font_size'] * s)
            if e.get('letter_spacing'):
                e['letter_spacing'] *= s
            for run in e.get('style_runs') or []:
                if isinstance(run.get('font_size'), (int, float)):
                    run['font_size'] = max(1.0, run['font_size'] * s)
        for key in ('stroke_width', 'corner_radius'):
            if isinstance(e.get(key), (int, float)):
                e[key] *= s
    return out


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


def _save_render(design, folder, name, notes=None):
    """Render with the official renderer; returns the PNG file name, or None.

    A design whose only problem is a missing brand-kit image (the challenge's Level 3 sample
    does this on purpose) is drawn with the renderer's labelled placeholder instead, and the
    missing files are added to `notes`.
    """
    result = verify_design(design)
    image = result.get('image')
    if not result.get('verified') or image is None:
        fallback = unverified_missing_asset_preview(design)
        image = fallback.get('image')
        if image is None:
            return None
        if notes is not None:
            notes.update(fallback.get('missing_assets', []))
    image.save(folder / f'{name}.png', format='PNG')
    return f'{name}.png'


def _missing_note(notes):
    return ('Missing image shown as a placeholder: ' + ', '.join(sorted(notes))) if notes else None


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
        notes = set()
        t0 = time.monotonic()
        try:
            if same_shape(source, target):
                design = scale_design(source, target)
                (folder / f'{i}.json').write_text(json.dumps(design, indent=2))
                png = _save_render(design, folder, str(i), notes)
                if png is None:
                    item.update(state='failed', error='The renderer could not draw this layout.')
                    continue
                exact = (target['width'], target['height']) == (source['canvas']['width'], source['canvas']['height'])
                item.update(state='done', render=png, design=f'{i}.json', method='original' if exact else 'scaled', warning=_missing_note(notes))
                continue
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
            png = _save_render(best['design'], folder, str(i), notes)
            if png is None:
                item.update(state='failed', error='The renderer could not draw this layout.')
                continue
            item.update(state='done', render=png, design=f'{i}.json', method=best.get('template', ''), warning=_missing_note(notes))
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
    notes = set()
    png = _save_render(req.design, folder, name, notes)
    if png is None:
        raise HTTPException(422, 'The renderer could not draw this design (missing font or unreadable image?).')
    return {'render': f'/files/previews/{png}', 'warning': _missing_note(notes)}


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
