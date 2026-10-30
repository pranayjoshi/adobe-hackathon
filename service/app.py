"""Job API wrapping the engine for the viewer.

Run from the repo root:
    .venv/bin/uvicorn service.app:app --reload --port 8765
"""

from __future__ import annotations

import hashlib
import json
import time
import math
import threading
import subprocess
import sys
import os
import signal
import copy
import tempfile
from engine.quality import _RENDER_LOCK
from service.workspace import router, save_run, POOL, JOBS, LOCK
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from service.workspace import repository
from service.storage import Conflict
from engine.contracts import NoValidDesign, design_hash

from engine.analysis import analyze
from engine.canonical import ASSETS_DIR, ROOT
from engine.invariants import validate
from engine.pipeline import generate, uniform_stretch
from engine.typography import renderer_available

INPUT_DIR = ROOT / "input"
RENDER_DIR = ROOT / "output" / "renders"
RENDER_DIR.mkdir(parents=True, exist_ok=True)
_WORKER_CANCEL_GRACE_SECONDS = 15.0

try:
    from renderer.converter.convert import render_design_to_png
except Exception:  # pragma: no cover
    render_design_to_png = None

app = FastAPI(title="Chameleon", version="0.2.0")
app.include_router(router)
from service.crello import router as crello_router
app.include_router(crello_router)
from service.judging import router as judge_router
app.include_router(judge_router)
from service.express import router as express_router
app.include_router(express_router)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5180","http://127.0.0.1:5180","http://localhost:8765","http://127.0.0.1:8765","http://127.0.0.1:5241","https://localhost:5241"], allow_methods=["*"], allow_headers=["*"])
app.mount("/renders", StaticFiles(directory=str(RENDER_DIR)), name="renders")

# Target catalogue: real platform formats, described the way a designer would.
PRESETS = [
    {"id": "ig-post", "group": "Social", "label": "Instagram Post", "width": 1080, "height": 1080},
    {"id": "ig-portrait", "group": "Social", "label": "Instagram Portrait", "width": 1080, "height": 1350},
    {"id": "ig-story", "group": "Social", "label": "Story / Reel", "width": 1080, "height": 1920},
    {"id": "yt-thumb", "group": "Social", "label": "YouTube Thumbnail", "width": 1280, "height": 720},
    {"id": "x-post", "group": "Social", "label": "X / Twitter Post", "width": 1600, "height": 900},
    {"id": "li-banner", "group": "Social", "label": "LinkedIn Banner", "width": 1584, "height": 396},
    {"id": "fb-cover", "group": "Social", "label": "Facebook Cover", "width": 1640, "height": 624},
    {"id": "pin", "group": "Social", "label": "Pinterest Pin", "width": 1000, "height": 1500},
    {"id": "web-hero", "group": "Web", "label": "Website Hero", "width": 1600, "height": 400},
    {"id": "leaderboard", "group": "Display ads", "label": "Leaderboard", "width": 728, "height": 90},
    {"id": "billboard", "group": "Display ads", "label": "Billboard", "width": 970, "height": 250},
    {"id": "mrec", "group": "Display ads", "label": "Medium Rectangle", "width": 300, "height": 250},
    {"id": "skyscraper", "group": "Display ads", "label": "Wide Skyscraper", "width": 160, "height": 600},
    {"id": "half-page", "group": "Display ads", "label": "Half Page", "width": 300, "height": 600},
]



def _render(design: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Resource-keyed isolated official preview; a native crash cannot kill API."""
    from engine.quality import verify_design,resource_fingerprint
    try:
        identity=design_hash(design);resources=resource_fingerprint(design)
        key=hashlib.sha256((identity+resources).encode()).hexdigest()[:24]
        cached=RENDER_DIR/(key+'.png');sidecar=cached.with_suffix('.json')
        if cached.is_file() and sidecar.is_file():
            metadata=json.loads(sidecar.read_text())
            if metadata.get('verified') and metadata.get('design_hash')==identity and metadata.get('resource_hash')==resources:
                from PIL import Image
                with Image.open(cached) as image:
                    if image.size==(int(design['canvas']['width']),int(design['canvas']['height'])):
                        image.verify();return '/renders/'+cached.name,[]
    except (OSError,ValueError,KeyError):pass
    result=verify_design(design)
    image=result.pop('image',None)
    if not result.get('verified') or image is None:
        return None,[result.get('error','Official preview unavailable')]
    key=hashlib.sha256((result['design_hash']+result.get('resource_hash','')).encode()).hexdigest()[:24]
    path=RENDER_DIR/(key+'.png')
    # A successful fast-path cache hit already returned. If verification was
    # needed, replace even an existing corrupt PNG rather than blessing it.
    with tempfile.NamedTemporaryFile(dir=RENDER_DIR,suffix='.png',delete=False) as handle:temp=Path(handle.name)
    try:image.save(temp,format='PNG');temp.replace(path)
    finally:temp.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=RENDER_DIR,suffix='.json',mode='w',delete=False) as handle:
        json.dump(result,handle,allow_nan=False);pending=Path(handle.name)
    try:pending.replace(path.with_suffix('.json'))
    finally:pending.unlink(missing_ok=True)
    return '/renders/'+path.name,result.get('warnings',[])


def _unverified_source_preview(design: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Show a source whose only official failure is a missing original image.

    The file is a display placeholder. Its sidecar stays unverified, so the
    official cache will not treat those pixels as a successful render.
    """
    from engine.quality import resource_fingerprint, unverified_missing_asset_preview
    try:
        resources=resource_fingerprint(design, allow_missing=True)
    except (ValueError, OSError) as exc:
        return None, [str(exc)]
    identity=design_hash(design)
    key=hashlib.sha256(('unverified-missing-asset-v1'+identity+resources).encode()).hexdigest()[:24]
    path=RENDER_DIR/('unverified-'+key+'.png')
    sidecar=path.with_suffix('.json')
    if path.is_file() and sidecar.is_file():
        try:
            metadata=json.loads(sidecar.read_text())
            if metadata.get('verified') is False and metadata.get('counts_as_success') is False and metadata.get('design_hash')==identity and metadata.get('resource_hash')==resources:
                from PIL import Image
                with Image.open(path) as image:
                    if image.size==(int(design['canvas']['width']),int(design['canvas']['height'])):
                        image.verify();return '/renders/'+path.name, list(metadata.get('warnings') or [])
        except (OSError, ValueError, KeyError):
            pass
    result=unverified_missing_asset_preview(design)
    image=result.pop('image', None)
    if image is None or result.get('verified') or result.get('counts_as_success') is not False:
        return None, [result.get('error', 'Official preview unavailable')]
    with tempfile.NamedTemporaryFile(dir=RENDER_DIR, suffix='.png', delete=False) as handle:
        temp=Path(handle.name)
    try:
        image.save(temp, format='PNG');temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(dir=RENDER_DIR, suffix='.json', mode='w', delete=False) as handle:
        json.dump(result, handle, allow_nan=False);pending=Path(handle.name)
    try:
        pending.replace(sidecar)
    finally:
        pending.unlink(missing_ok=True)
    return '/renders/'+path.name, list(result.get('warnings') or [])


def _missing_asset_failure(warnings: list[str]) -> bool:
    return bool(warnings) and all(str(item).startswith(('Missing or unsafe image asset:', "Missing asset '")) for item in warnings)


def _display_render(design: dict[str, Any]) -> tuple[str | None, list[str], bool]:
    """Official preview when it verifies; otherwise a labeled placeholder for a missing image.

    A placeholder URL is never verified. Other renderer failures stay hidden.
    """
    url, warnings = _render(design)
    if url:
        return url, warnings, True
    if _missing_asset_failure(warnings):
        diagnostic, diagnostic_warnings = _unverified_source_preview(design)
        if diagnostic:
            return diagnostic, diagnostic_warnings, False
    return None, warnings, False


def _preview_verification(design, url, verified=None):
    from engine.quality import resource_fingerprint, VERIFICATION_PROFILE
    if verified is None:
        verified = url is not None
    resource = None
    if url:
        try:
            resource = resource_fingerprint(design, allow_missing=not verified)
        except (ValueError, OSError):
            verified = False
    return {'verified': bool(verified and url), 'design_hash': design_hash(design),
            'resource_hash': resource, 'profile': VERIFICATION_PROFILE if verified else 'unverified-missing-asset-v1'}


class Target(BaseModel):
    width: int = Field(gt=0, le=8000)
    height: int = Field(gt=0, le=8000)
    label: str | None = None
    id: str | None = None


class GenerateRequest(BaseModel):
    source: dict[str, Any]
    targets: list[Target] = Field(min_length=1, max_length=20)
    emphasis: str = "balanced"
    top_k: int = Field(default=3, ge=1, le=36)
    locks: dict[str, dict[str,float]] = Field(default_factory=dict)
    analysis_overrides: dict[str,dict[str,str]] = Field(default_factory=dict)
    options: dict[str, bool | str | int | float] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None,max_length=200)

    @field_validator('analysis_overrides',mode='before')
    @classmethod
    def valid_analysis_overrides(cls,value):
        if not isinstance(value,dict) or set(value)-{'roles','group_modes'}:
            raise ValueError('Intent supports role and group-mode sidecars')
        if any(not isinstance(v,dict) or len(v)>200 or any(not isinstance(k,str) or not isinstance(x,str) for k,x in v.items()) for v in value.values()):
            raise ValueError('Intent corrections must map IDs to supported string values')
        return value

    @field_validator('locks',mode='before')
    @classmethod
    def numeric_locks(cls,value):
        if not isinstance(value,dict) or any(not isinstance(patch,dict) for patch in value.values()):raise ValueError('Locks must be element geometry objects')
        for patch in value.values():
            if any(isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) for number in patch.values()):raise ValueError('Locks must use finite numbers, never booleans')
        return value

    @field_validator('options',mode='before')
    @classmethod
    def valid_options(cls,value):
        if not isinstance(value,dict):raise ValueError('options must be an object')
        boolean={'solver','render','refine','retrieval','vlm','roles','pixel_diagnostics','learned_edit','learned_retarget'}
        choices={'engine':{'final','legacy'},'controller_mode':{'legacy_batch','single','single_restart','small_beam'},
                 'solver_mode':{'legacy','joint'},'visual_judge':{'off','openjev'},
                 'group_policies':{'legacy','hierarchical'},'role_mode':{'legacy','shadow','calibrated'},
                 'proposal_portfolio':{'legacy','diverse'},'retrieval_version':{'legacy','fit_aware'},
                 'refinement_mode':{'hill_climb','beam_lns'},'photo_mode':{'heuristic','subject_aware'},
                 'background_mode':{'legacy','safe_cover'}}
        limits={'controller_seconds':(.05,300),'controller_target':(0,100),
                'max_edits':(1,500),'max_renders':(1,100),'max_model_calls':(0,100)}
        for key,item in value.items():
            if key in boolean:
                if not isinstance(item,bool):raise ValueError(key+' must be boolean')
            elif key in choices:
                if not isinstance(item,str) or item not in choices[key]:raise ValueError('Unknown '+key)
            elif key in limits:
                low,high=limits[key]
                if isinstance(item,bool) or not isinstance(item,(int,float)) or not math.isfinite(item) or not low<=item<=high:raise ValueError('Invalid '+key)
                if key.startswith('max_') and int(item)!=item:raise ValueError(key+' must be an integer')
                if key.startswith('max_'):value[key]=int(item)
            else:raise ValueError('Unknown generation option: '+key)
        return value
    all_candidates: bool = False


class DesignRequest(BaseModel):
    design: dict[str, Any]
    group_policies: Literal['legacy','hierarchical'] | None = None


def _check_source(source: dict[str, Any]) -> None:
    if "canvas" not in source or "elements" not in source:
        raise HTTPException(422, "Design must have 'canvas' and 'elements'.")
    canvas=source["canvas"]
    if not isinstance(canvas,dict) or any(not isinstance(canvas.get(k),(int,float)) or isinstance(canvas.get(k),bool) or not math.isfinite(canvas[k]) or not 1<=canvas[k]<=8000 for k in ("width","height")):
        raise HTTPException(422,"Canvas dimensions must be finite and between 1 and 8000.")
    if not isinstance(source["elements"],list) or not 1<=len(source["elements"])<=200:
        raise HTTPException(422,"Expected between 1 and 200 elements.")
    if any(not isinstance(e,dict) for e in source['elements']):raise HTTPException(422,'Elements must be objects')
    ids = [e.get("id") for e in source["elements"]]
    if any(not isinstance(i,str) for i in ids):raise HTTPException(422,"IDs must be strings.")
    if len(ids) != len(set(ids)) or None in ids:
        raise HTTPException(422, "Every element needs a unique 'id'.")
    for e in source["elements"]:
        required={"text":("content","font_family","font_size","color"),"image":("asset_id",),"shape":("shape_type","fill_color")}
        if e.get('type') not in required or any(k not in e for k in required[e['type']]):raise HTTPException(422,'Missing required element fields')
        if e['type']=='image':
            if not isinstance(e['asset_id'],str) or not (ASSETS_DIR/e['asset_id']).resolve().is_relative_to(ASSETS_DIR.resolve()):raise HTTPException(422,'Asset references must stay inside brand_kit/assets')
        if e['type']=='text':
            if not isinstance(e['content'],str) or len(e['content'])>20000 or not isinstance(e['font_family'],str) or isinstance(e['font_size'],bool) or not isinstance(e['font_size'],(int,float)) or not math.isfinite(e['font_size']) or not 1<=e['font_size']<=10000:raise HTTPException(422,'Invalid text content, font or size')
        for k in ("x", "y", "width", "height"):
            if not isinstance(e.get(k), (int, float)) or isinstance(e.get(k),bool) or not math.isfinite(e[k]):
                raise HTTPException(422, f"Element '{e.get('id')}' is missing numeric '{k}'.")
            if k in ('width','height') and e[k]<=0:raise HTTPException(422,'Element dimensions must be positive')
            if abs(e[k])>32768:raise HTTPException(422,'Element geometry exceeds the local resource limit')
        if e['width']*e['height']>32000000:raise HTTPException(422,'Element pixel area exceeds the local resource limit')
        for key,lo,hi in (('rotation',-36000,36000),('z_index',-1000000,1000000),('opacity',0,1),('letter_spacing',-1000,1000),('line_height',.1,10)):
            if key in e and (isinstance(e[key],bool) or not isinstance(e[key],(int,float)) or not math.isfinite(e[key]) or not lo<=e[key]<=hi):raise HTTPException(422,'Invalid element '+key)
        if 'style_runs' in e:
            runs=e['style_runs']
            if not isinstance(runs,list) or len(runs)>1000:raise HTTPException(422,'Invalid style runs')
            for run in runs:
                if not isinstance(run,dict):raise HTTPException(422,'Style runs must be objects')
                for key in ('start','end'):
                    if key in run and (isinstance(run[key],bool) or not isinstance(run[key],int) or not 0<=run[key]<=100000):raise HTTPException(422,'Invalid style-run index')
                if run.get('end',100000)<run.get('start',0):raise HTTPException(422,'Style-run end precedes its start')
                for key in ('font_weight','font_style','color'):
                    if key in run and not isinstance(run[key],str):raise HTTPException(422,'Invalid style-run '+key)
                if 'italic' in run and not isinstance(run['italic'],bool):raise HTTPException(422,'Invalid style-run italic')
        for key in ('font_weight','font_style','text_align'):
            if key in e and not isinstance(e[key],str):raise HTTPException(422,'Invalid element '+key)
    if sum(len(e.get('content','')) for e in source['elements'] if e['type']=='text')>100000:raise HTTPException(422,'Total text exceeds the local resource limit')


@app.get("/api/health")
def health() -> dict[str, Any]:
    from engine.retarget.propose import available as learned_layouts
    return {"ok": True, "renderer": render_design_to_png is not None, "text_metrics": renderer_available(),
            "learned_layouts": learned_layouts()}


@app.get("/api/presets")
def presets() -> list[dict[str, Any]]:
    return PRESETS


@app.get("/api/samples")
def samples() -> list[dict[str, Any]]:
    out = []
    for p in sorted(INPUT_DIR.glob("*.json")):
        t = json.loads(p.read_text())
        out.append({
            "id": p.stem,
            "title": t.get("difficulty_label") or p.stem,
            "notes": t.get("notes", ""),
            "source": t["source"],
            "target_canvas": t.get("target_canvas"),
        })
    return out


@app.post("/api/analyze")
def analyze_design(req: DesignRequest) -> dict[str, Any]:
    _check_source(req.design)
    url, warnings = _render(req.design)
    preview_verified=url is not None
    if url is None:
        diagnostic, diagnostic_warnings=_unverified_source_preview(req.design)
        if diagnostic:
            url, warnings, preview_verified=diagnostic, diagnostic_warnings, False
    return {"analysis": analyze(req.design,group_policies=req.group_policies).to_dict(), "render": url, "warnings": warnings, "preview_verified": preview_verified}


@app.post("/api/render")
def render(req: DesignRequest) -> dict[str, Any]:
    _check_source(req.design)
    url, warnings = _render(req.design)
    return {"render": url, "warnings": warnings}


def _validate_request(req):
    _check_source(req.source)
    known={e['id'] for e in req.source['elements']}
    if set(req.locks)-known:raise HTTPException(422,'Lock refers to an unknown element')
    for rect in req.locks.values():
        if set(rect)-{'x','y','width','height'} or not {'x','y'}<=set(rect) or any(not math.isfinite(v) or v<0 for v in rect.values()):raise HTTPException(422,'Invalid locked rectangle')
    if req.analysis_overrides:
        try:analyze(req.source,group_policies=req.options.get('group_policies'),overrides=req.analysis_overrides)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc


def _execute(req, rid, *, cancelled=None, target_offset=0, persist=True,owner=None):
    """One parent run with per-target controller events and revision artifacts."""
    _validate_request(req)
    t0=time.monotonic();A=analyze(req.source,group_policies=req.options.get('group_policies'),
        overrides=req.analysis_overrides or None);results=[];repo=repository()
    for local_index,target in enumerate(req.targets):
        index=target_offset+local_index
        if cancelled is not None and cancelled():break
        tgt={'width':target.width,'height':target.height};tg0=time.monotonic()
        def progress(event):
            repo.append_event(rid,{**event,'target_index':index},owner)
            repo.update_job(rid,{'stage':event.get('stage',event.get('type','evaluating'))},owner)
        def artifact(event):
            if not repo.owns(rid,owner):return
            event=dict(event);image=event.pop('image',None)
            hashed=design_hash(event['design']);folder=repo.root/rid/str(index)
            if owner:folder=folder/owner;event['artifact_attempt']=owner
            folder.mkdir(parents=True,exist_ok=True)
            json_path=folder/(hashed+'.json');json_path.write_text(json.dumps(event['design'],allow_nan=False))
            if image is not None:
                image_path=folder/(hashed+'.png');pending=folder/(hashed+'.tmp.png')
                image.save(pending,format='PNG');pending.replace(image_path)
                event['render']=f'/api/history/{rid}/targets/{index}/revisions/{hashed}/image'
            accepted=bool(event.get('accepted') and event.get('verified') and image is not None)
            saved=repo.put_revision(rid,index,event,activate=accepted,owner=owner)
            if saved is None:return
            if accepted:
                preview={'render':event['render'],'revision_hash':hashed,'target_index':index,
                         'width':tgt['width'],'height':tgt['height'],'operation':event.get('operation')}
                repo.update_job(rid,{'active_preview':preview},owner)
                repo.append_event(rid,{'type':'preview_ready','target_index':index,'revision_hash':hashed,'render':event['render']},owner)
        try:
            if req.options.get('engine')=='final' and not req.locks:
                from final.resize import ranked
                cands=ranked(req.source,tgt)[0][:req.top_k]
                if not cands:raise NoValidDesign('No layout could be generated')
            else:
                cands=generate(req.source,tgt,req.emphasis,req.top_k,analysis=A,locks=req.locks,
                    options=req.options,all_candidates=req.all_candidates,progress_callback=progress,
                    artifact_callback=artifact,cancelled=cancelled)
        except NoValidDesign as exc:
            results.append({'target':{**tgt,'label':target.label,'id':target.id},'candidates':[],
                'baseline':{'design':uniform_stretch(req.source,tgt),'render':None,'checks':[]},
                'status':'no_valid_design','error':str(exc),'controller':exc.report,
                'elapsed_ms':round((time.monotonic()-tg0)*1000)})
            continue
        failed_candidates=[];verified_candidates=[]
        for position,candidate in enumerate(cands):
            candidate['revision_hash']=design_hash(candidate['design'])
            existing=repo.revision(rid,index,candidate['revision_hash']) or {}
            url=existing.get('render');warnings=[]
            if url:verified=bool(existing.get('verified', True))
            else:url,warnings,verified=_display_render(candidate['design'])
            usable=bool(verified and candidate['valid'] and candidate.get('rubric',{}).get('gate',True))
            candidate.update(render=url,render_warnings=warnings,preview_verified=verified,exportable=usable,
                             verification={**candidate.get('verification',{}),**_preview_verification(candidate['design'],url,verified)})
            saved={**existing,'design':candidate['design'],'candidate':candidate,'render':url,
                   'accepted':usable and not verified_candidates,'verified':verified,
                   'resource_hash':candidate['verification'].get('resource_hash')}
            repo.put_revision(rid,index,saved,activate=usable and not verified_candidates,owner=owner)
            (verified_candidates if usable else failed_candidates).append(candidate)
        cands=verified_candidates
        status='done' if cands else 'no_verified_output'
        if not cands:
            visible=[c for c in failed_candidates if c.get('render') and c.get('preview_verified') is False and _missing_asset_failure(c.get('render_warnings') or [])]
            if visible:
                cands=visible
                status='unverified_missing_asset'
        baseline=uniform_stretch(req.source,tgt);baseline_url,baseline_warnings,baseline_verified=_display_render(baseline)
        results.append({'target':{**tgt,'label':target.label,'id':target.id},'candidates':cands,
            'baseline':{'design':baseline,'render':baseline_url,'checks':validate(req.source,baseline),'warnings':baseline_warnings,'preview_verified':baseline_verified},
            'elapsed_ms':round((time.monotonic()-tg0)*1000),'status':status,
            'failed_candidates':failed_candidates})
    response={'analysis':A.to_dict(),'results':results,'elapsed_ms':round((time.monotonic()-t0)*1000),'run_id':rid}
    if persist:save_run(req.model_dump(),response,rid,owner)
    return response


@app.post('/api/generate')
def generate_layouts(req:GenerateRequest):
    """Synchronous compatibility endpoint using the same isolated durable job."""
    started=start_job(req);rid=started['id'];repo=repository()
    while True:
        job=repo.job(rid)
        if job['state'] not in ('queued','running'):
            response=job.get('result') or repo.run(rid)['response']
            if job['state']=='done':return response
            raise HTTPException(422 if job['state']=='failed' else 409,{**response,'state':job['state'],'error':job.get('error')})
        time.sleep(.05)


def _perform_job(req,rid,owner,repo):
    """Worker entry: all native engine/renderer operations stay in this process."""
    def is_cancelled():
        current=repo.job(rid)
        return current is None or current.get('cancel',False) or current.get('state')!='running' or current.get('owner')!=owner
    prior=repo.run(rid)
    responses=copy.deepcopy(prior['response']['results']) if prior else []
    t0=time.monotonic();repo.append_event(rid,{'type':'job_started','stage':'initializing'},owner)
    for index,target in enumerate(req.targets):
        if is_cancelled():break
        if index<len(responses) and responses[index].get('status')=='done':continue
        response=_execute(req.model_copy(update={'targets':[target]}),rid,cancelled=is_cancelled,
                          target_offset=index,persist=False,owner=owner)
        current=repo.job(rid)
        if not current or current.get('owner')!=owner or current['state']!='running':return
        if not response['results']:break
        if index<len(responses):responses[index]=response['results'][0]
        else:responses.extend(response['results'])
        combined={'analysis':response['analysis'],'results':responses,'elapsed_ms':round((time.monotonic()-t0)*1000),'run_id':rid}
        save_run(req.model_dump(),combined,rid,owner)
        repo.update_job(rid,{'progress':len(responses)/len(req.targets),'result':combined},owner)
    state='cancelled' if is_cancelled() else 'done'
    if state=='done' and responses and all(not result['candidates'] for result in responses):state='failed'
    repo.append_event(rid,{'type':'job_stopped','state':state},owner)
    repo.update_job(rid,{'state':state,'stage':state,'elapsed_ms':round((time.monotonic()-t0)*1000)},owner)


def _supervise_job(req,rid,owner,repo):
    """Fresh interpreter avoids forking a process with loaded Skia state.

    The process group includes isolated renderer children. Cooperative cancel
    gets 15 seconds, then the entire owned group is terminated and reaped.
    """
    folder=repo.root/rid;folder.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+len(req.targets)*max(180 if req.options.get('engine')=='final' else 45,float(req.options.get('controller_seconds',12))+20)
    cancelled_at=None;reason=None
    with (folder/'worker.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-m','service.job_worker',rid,owner,str(repo.root)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        repo.update_job(rid,{'worker_pid':process.pid},owner)
        try:
            while process.poll() is None:
                current=repo.job(rid)
                retired=not current or current.get('owner')!=owner or current['state']=='interrupted'
                if retired:
                    reason='Worker ownership retired';break
                if current.get('cancel'):
                    if cancelled_at is None:cancelled_at=time.monotonic()
                    if time.monotonic()-cancelled_at>=_WORKER_CANCEL_GRACE_SECONDS:
                        reason='Cancellation grace period expired';break
                if time.monotonic()>=deadline:
                    reason='Worker exceeded its declared execution deadline';break
                time.sleep(.05)
            if reason:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait(timeout=5)
            else:process.wait()
        finally:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=5)
            # The leader can exit on TERM while its renderer grandchildren
            # ignore it. Always clean the owned group independently of poll().
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    current=repo.job(rid)
    if current and current['state'] in ('queued','running'):
        state='cancelled' if current.get('cancel') else 'failed'
        error=reason or ('Isolated worker exited with code '+str(process.returncode))
        repo.append_event(rid,{'type':'worker_stopped','state':state,'error':error},owner)
        repo.update_job(rid,{'state':state,'stage':state,'error':error,'worker_exit_code':process.returncode},owner)


def _submit_job(req,rid):
    repo=repository();owner=uuid.uuid4().hex
    with LOCK:JOBS[rid]={'id':rid,'cancel':False,'state':'queued'}
    def work():
        if not repo.claim(rid,owner):return
        with LOCK:JOBS[rid]['state']='running'
        try:_supervise_job(req,rid,owner,repo)
        except Exception as exc:
            repo.append_event(rid,{'type':'job_failed','error':type(exc).__name__+': '+str(exc)},owner)
            repo.update_job(rid,{'state':'failed','stage':'failed','error':type(exc).__name__+': '+str(exc)},owner)
        finally:
            current=repo.job(rid)
            with LOCK:JOBS[rid]['state']=current['state'] if current else 'failed'
    POOL.submit(work)


@app.post('/api/jobs')
def start_job(req:GenerateRequest):
    _validate_request(req)
    with LOCK:
        repo=repository();repo.interrupt_orphaned(JOBS.keys())
        try:rid,created=repo.create_job(uuid.uuid4().hex[:16],req.model_dump(),req.idempotency_key)
        except Conflict as exc:raise HTTPException(409 if req.idempotency_key else 429,str(exc))
        if created:
            empty={'canvas':req.source['canvas'],'roles':{},'blocks':[],'stickers':[],'accents':[],
                   'background':[],'decorations':[],'notes':[],'panels':[],'relationships':[]}
            save_run(req.model_dump(),{'analysis':empty,'results':[],'elapsed_ms':0,'run_id':rid},rid)
            _submit_job(req,rid)
    return {'id':rid,'run_id':rid,'created':created}


@app.post('/api/jobs/{rid}/resume')
def resume_job(rid:str):
    from service.workspace import safe_id
    safe_id(rid);repo=repository();request=repo.job_request(rid)
    if request is None:raise HTTPException(404,'Job not found')
    with LOCK:
        if not repo.requeue(rid):raise HTTPException(409,'Only interrupted, cancelled or failed jobs can resume')
        repo.append_event(rid,{'type':'job_resumed','stage':'queued'})
        _submit_job(GenerateRequest.model_validate(request),rid)
    return {'id':rid,'run_id':rid,'resumed':True}


@app.post('/api/express/assets')
def express_assets(req: DesignRequest):
    """Prepare original images at their layout size for supported Express import."""
    import base64,io
    from PIL import Image,ImageOps
    _check_source(req.design)
    assets={}
    for e in req.design['elements']:
        if e.get('type')!='image':continue
        path=(ASSETS_DIR/str(e.get('asset_id',''))).resolve()
        if not path.is_relative_to(ASSETS_DIR.resolve()) or not path.is_file():raise HTTPException(422,'Original image unavailable: '+e['id'])
        width,height=max(1,round(e['width'])),max(1,round(e['height']))
        if width*height>16000000:raise HTTPException(422,'Asset exceeds local preparation limit')
        with Image.open(path) as im:
            im=im.convert('RGBA')
            if e.get('fit')=='cover':prepared=ImageOps.fit(im,(width,height))
            elif e.get('fit')=='fill':prepared=im.resize((width,height))
            else:
                fitted=ImageOps.contain(im,(width,height));prepared=Image.new('RGBA',(width,height));prepared.alpha_composite(fitted,((width-fitted.width)//2,(height-fitted.height)//2))
            output=io.BytesIO();prepared.save(output,format='PNG');assets[e['id']]=base64.b64encode(output.getvalue()).decode()
    return {'assets':assets}


# Serve the built viewer if present (npm run build in viewer/).
_dist = Path(ROOT / "viewer" / "dist")
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="viewer")
