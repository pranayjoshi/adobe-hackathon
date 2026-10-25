"""Local run history, blind pair labeling, asynchronous jobs, and eval browsing."""
import copy,hashlib,json,random,threading,time,uuid,math
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field,field_validator
from typing import Literal
from engine.contracts import design_hash
from .storage import Repository,Conflict
from engine.canonical import ROOT
from engine.learning import RANK_NAMES
from engine.vlm import status as vlm_status
from engine.constitution import CRITERIA
router=APIRouter();STORE=ROOT/'output/workspace';STORE.mkdir(parents=True,exist_ok=True)
LOCK=threading.RLock();POOL=ThreadPoolExecutor(max_workers=1);JOBS={}

def repository():
    return Repository(STORE)

def labelable(candidates):
    """Only candidates passing rubric Tier 0-1 are shown to annotators (training uses gated pairs only)."""
    return [c for c in candidates if c['valid'] and c.get('render') and c.get('preview_verified',True)
            and (c.get('verification') or {}).get('verified',True) and (c.get('rubric') or {}).get('gate',True)]

def safe_id(value):
    if not value or not all(c.isalnum() or c in '-_' for c in value):raise HTTPException(422,'Invalid record ID')
    return value

def save_run(request,response,rid=None,owner=None):
    rid=rid or uuid.uuid4().hex[:16];safe_id(rid);source=request['source'];family=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()[:16]
    repo=repository();previous=repo.run(rid)
    if previous:resource_hash=previous.get('resource_hash')
    else:
        try:
            from engine.quality import resource_fingerprint
            resource_hash=resource_fingerprint(source)
        except (ValueError,OSError):resource_hash=None
    response['run_id']=rid
    record={'id':rid,'family':family,'resource_hash':resource_hash,'created':previous['created'] if previous else time.time(),'request':request,'response':copy.deepcopy(response)}
    with LOCK:
        if not repo.put_run(record,owner):raise Conflict('Worker claim expired before run persistence')
        path=STORE/(rid+'.json')
        with tempfile.NamedTemporaryFile(dir=STORE,suffix='.json',mode='w',delete=False) as handle:
            json.dump(record,handle,allow_nan=False);tmp=Path(handle.name)
        try:tmp.replace(path)
        finally:tmp.unlink(missing_ok=True)
    return rid

@router.get('/api/history')
def history():
    records={r['id']:r for r in repository().runs()}
    for p in STORE.glob('*.json'):
        if p.name=='labels.json':continue
        try:
            r=json.loads(p.read_text());records.setdefault(r['id'],r)
        except (KeyError,ValueError):continue
    result=[{'id':r['id'],'family':r['family'],'created':r['created'],'targets':len(r['response']['results'])} for r in records.values()]
    return sorted(result,key=lambda x:-x['created'])[:100]

@router.get('/api/history/{rid}')
def record(rid:str):
    saved=repository().run(safe_id(rid))
    if saved:return saved
    p=STORE/(safe_id(rid)+'.json')
    if not p.is_file():raise HTTPException(404,'Run not found')
    saved=json.loads(p.read_text());repository().put_run(saved);return saved

@router.get('/api/capabilities')
def capabilities():
    return {'solver':True,'solver_modes':['legacy','joint'],'controller_modes':['legacy_batch','single','single_restart','small_beam'],'durable_jobs':True,'revision_feedback':True,'edit_constraints':True,'render_repair':True,'retrieval_library':(ROOT/'engine/models/layouts.json').exists(),'role_weights':(ROOT/'engine/models/roles.json').exists(),'rank_weights':(ROOT/'engine/models/ranker.json').exists(),'vlm':vlm_status(),'ranking_features':RANK_NAMES,'criteria':CRITERIA}

@router.get('/api/preferences/pair')
def pair():
    runs=history()
    with LOCK:
        labels=STORE/'labels.jsonl';seen={x['pair'] for x in (json.loads(l) for l in labels.read_text().splitlines())} if labels.exists() else set()
    rng=random.Random(len(seen))
    rng.shuffle(runs)
    for meta in runs:
        r=record(meta['id'])
        for index,target in enumerate(r['response']['results']):
            cs=labelable(target['candidates'])
            for i in range(len(cs)):
                for j in range(i+1,len(cs)):
                    key=f'{r["id"]}-{index}-{i}-{j}'
                    if key in seen:continue
                    choices=[i,j];rng.shuffle(choices)
                    return {'pair':key,'run_id':r['id'],'target_index':index,'a_index':choices[0],'b_index':choices[1],'family':r['family'],'source':r['request']['source'],'source_hash':design_hash(r['request']['source']),'a':{'render':cs[choices[0]].get('render'),'design':cs[choices[0]]['design']},'b':{'render':cs[choices[1]].get('render'),'design':cs[choices[1]]['design']},'target':target['target']}
    return None

class Label(BaseModel):
    pair:str;run_id:str;target_index:int=Field(ge=0);a_index:int=Field(ge=0);b_index:int=Field(ge=0);decision:str
    reason:str|None=None;defects:list[str]=Field(default_factory=list)

@router.post('/api/preferences')
def label(req:Label):
    if req.decision not in ('a','b','tie','both_bad'):raise HTTPException(422,'Invalid verdict')
    if (req.reason and req.reason not in CRITERIA) or set(req.defects)-set(CRITERIA):raise HTTPException(422,'Unknown criterion')
    r=record(req.run_id)
    try:
        cs=labelable(r['response']['results'][req.target_index]['candidates']);a,b=cs[req.a_index],cs[req.b_index]
        if req.a_index==req.b_index:raise IndexError
    except IndexError:raise HTTPException(422,'Invalid pair')
    key=f'{req.run_id}-{req.target_index}-{min(req.a_index,req.b_index)}-{max(req.a_index,req.b_index)}'
    if req.pair!=key:raise HTTPException(422,'Pair mismatch')
    row={'pair':key,'family':r['family'],'kind':'human','provenance':'human','decision':req.decision,'reason':req.reason,'defects':req.defects,'a':a,'b':b,'a_hash':design_hash(a['design']),'b_hash':design_hash(b['design']),'created':time.time()}
    with LOCK:
        path=STORE/'labels.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        rows=[x for x in rows if x['pair']!=key]+[row];temp=path.with_suffix('.tmp');temp.write_text('\n'.join(json.dumps(x) for x in rows)+'\n');temp.replace(path)
    return {'saved':True,'pairs':len(rows)}

@router.get('/api/evaluation')
def evaluations():
    return [_evaluation_report(p)[1] for p in sorted((ROOT/'output/evaluation').glob('*/results.json'))]


def _evaluation_report(path):
    def reject_constant(value):raise ValueError('Non-finite JSON number: '+value)
    try:data=json.loads(path.read_text(),parse_constant=reject_constant)
    except (OSError,ValueError) as exc:
        return None,{'name':path.parent.name,'available':False,'status':'corrupt','diagnostic':str(exc)}
    if not isinstance(data,dict) or not isinstance(data.get('summary'),dict):
        return None,{'name':path.parent.name,'available':False,'status':'unsupported',
                     'diagnostic':'Expected an object containing an object summary; legacy list reports need conversion.'}
    summary=data['summary']
    for key in ('valid','cases'):
        if key in summary and (isinstance(summary[key],bool) or not isinstance(summary[key],int) or summary[key]<0):
            return None,{'name':path.parent.name,'available':False,'status':'unsupported',
                         'diagnostic':'Summary counts must be nonnegative integers.'}
    if 'valid' in summary and 'cases' in summary and summary['valid']>summary['cases']:
        return None,{'name':path.parent.name,'available':False,'status':'unsupported',
                     'diagnostic':'Valid count exceeds the complete case count.'}
    return data,{**summary,'name':path.parent.name,'available':True,'status':'available','summary':summary}

@router.get('/api/evaluation/{name}')
def evaluation(name:str):
    p=ROOT/'output/evaluation'/safe_id(name)/'results.json'
    if not p.is_file():raise HTTPException(404,'Evaluation not found')
    data,meta=_evaluation_report(p)
    if data is None:raise HTTPException(422,meta)
    return data

@router.get('/api/jobs/{rid}')
def job(rid:str,after_event:int=-1):
    with LOCK:
        repo=repository();value=repo.job(safe_id(rid))
        if value is None:raise HTTPException(404,'Job not found')
        if rid not in JOBS and value['state'] in ('queued','running'):
            repo.interrupt_orphaned(JOBS.keys());value=repo.job(rid)
        return {**value,'events':repo.events(rid,after_event)}

@router.post('/api/jobs/{rid}/cancel')
def cancel(rid:str):
    with LOCK:
        repo=repository();value=repo.job(safe_id(rid))
        if value is None:raise HTTPException(404,'Job not found')
        if value['state'] in ('queued','running'):
            repo.update_job(rid,{'cancel':True,'stage':'cancelling'})
            if rid in JOBS:JOBS[rid]['cancel']=True
            repo.append_event(rid,{'type':'cancellation_requested'})
    return {'requested':True,'detail':'Controller cancellation is checked between edits and solver operations. Active renderer calls have bounded timeouts. Partial results remain in this run.'}


class Feedback(BaseModel):
    target_index:int=Field(ge=0)
    revision_hash:str=Field(min_length=64,max_length=64)
    verdict:Literal['like','dislike','accept','export','a','b','tie','both_bad','correction']
    provenance:Literal['human','behavioral','model']='human'
    other_hash:str|None=None
    reason:str|None=Field(default=None,max_length=2000)
    idempotency_key:str|None=Field(default=None,max_length=200)
    reviewer_id:Literal['local']='local'
    resource_hash:str|None=None


@router.get('/api/history/{rid}/feedback')
def feedback_history(rid:str):
    record(rid)
    return repository().feedback_rows(rid)


@router.post('/api/history/{rid}/feedback')
def feedback(rid:str,req:Feedback):
    data=record(rid);repo=repository();revision=repo.revision(rid,req.target_index,req.revision_hash)
    if revision is None:
        raise HTTPException(409,'Feedback must identify an existing exact revision')
    if not revision.get('verified') or not revision.get('render'):
        raise HTTPException(422,'Visual feedback requires a verified rendered revision')
    if req.resource_hash is not None and req.resource_hash!=revision.get('resource_hash'):
        raise HTTPException(409,'The rendered resources changed. Reload before reviewing this design.')
    if req.other_hash:
        other=repo.revision(rid,req.target_index,req.other_hash)
        if other is None:raise HTTPException(409,'Comparison revision does not belong to this target')
        if not other.get('verified') or not other.get('render'):raise HTTPException(422,'Both comparison revisions require verified previews')
    if req.verdict in ('a','b','tie','both_bad') and (not req.other_hash or req.other_hash==req.revision_hash):
        raise HTTPException(422,'A comparison needs two distinct revisions')
    if req.verdict=='correction':
        raise HTTPException(422,'Commit a validated edit to record a correction demonstration')
    if req.verdict in ('accept','export') and req.provenance!='behavioral':
        raise HTTPException(422,'Acceptance and export are behavioral signals')
    try:
        saved=repo.feedback(rid,req.target_index,req.revision_hash,req.verdict,
            provenance=req.provenance,reason=req.reason,other_hash=req.other_hash,key=req.idempotency_key,
            reviewer_id=req.reviewer_id,context={'family':data['family'],'source_hash':design_hash(data['request']['source']),
                'target':revision['design']['canvas'],'options':data['request'].get('options',{}),
                'source_resource_hash':data.get('resource_hash'),'target_resource_hash':revision.get('resource_hash'),
                'verification':revision.get('assessment',{}).get('verification',{})})
    except Conflict as exc:raise HTTPException(409,str(exc))
    return {'saved':True,'feedback':saved}


@router.get('/api/history/{rid}/targets/{target_index}/revisions')
def revisions(rid:str,target_index:int):
    record(rid);repo=repository()
    return {'active_revision':repo.active_hash(rid,target_index),'revisions':repo.revisions(rid,target_index)}


@router.get('/api/history/{rid}/targets/{target_index}/revisions/{hashed}/image')
def revision_image(rid:str,target_index:int,hashed:str):
    repo=repository();safe_id(rid)
    if len(hashed)!=64 or any(c not in '0123456789abcdef' for c in hashed):raise HTTPException(422,'Invalid revision hash')
    revision=repo.revision(rid,target_index,hashed)
    if revision is None:raise HTTPException(404,'Revision not found')
    folder=repo.root/rid/str(target_index)
    if revision.get('artifact_attempt'):folder=folder/safe_id(revision['artifact_attempt'])
    path=folder/(hashed+'.png')
    if not path.is_file():raise HTTPException(404,'This revision was not rendered')
    return FileResponse(path,media_type='image/png')


class CommittedEdit(BaseModel):
    expected_revision_hash:str=Field(min_length=64,max_length=64)
    patches:dict[str,dict[str,float]]
    reason:str|None=Field(default=None,max_length=2000)

    @field_validator('patches',mode='before')
    @classmethod
    def finite_numeric_patches(cls,value):
        if not isinstance(value,dict) or any(not isinstance(patch,dict) for patch in value.values()):raise ValueError('Patches must be element geometry objects')
        for patch in value.values():
            if any(isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) for number in patch.values()):raise ValueError('Geometry must use finite numbers, never boolean values')
        return value


class RestoreRevision(BaseModel):
    expected_revision_hash:str=Field(min_length=64,max_length=64)
    revision_hash:str=Field(min_length=64,max_length=64)


@router.post('/api/history/{rid}/targets/{target_index}/edit')
def commit_edit(rid:str,target_index:int,req:CommittedEdit):
    from service.app import _render,_check_source,_preview_verification
    from engine.analysis import analyze
    from engine.pipeline import assess_design
    record_data=record(rid);repo=repository()
    current_job=repo.job(rid)
    if current_job and current_job['state'] in ('queued','running'):raise HTTPException(409,'Wait for this run to stop before committing a correction')
    current=repo.revision(rid,target_index,req.expected_revision_hash)
    if not current or repo.active_hash(rid,target_index)!=req.expected_revision_hash:
        raise HTTPException(409,'The active design changed. Reload before applying this edit.')
    design=copy.deepcopy(current['design']);by={e['id']:e for e in design['elements']}
    if not req.patches:raise HTTPException(422,'No geometry changes supplied')
    for eid,patch in req.patches.items():
        if eid not in by or not patch or set(patch)-{'x','y','width','height','font_size'}:
            raise HTTPException(422,'Only geometry and font size of existing elements may be edited')
        if any(not math.isfinite(value) or (key in ('width','height','font_size') and value<=0) for key,value in patch.items()):
            raise HTTPException(422,'Geometry must be finite with positive sizes')
        if 'font_size' in patch and by[eid]['type']!='text':raise HTTPException(422,'Font size belongs to text only')
        by[eid].update(patch)
    _check_source(design)
    source=record_data['request']['source']
    candidate=assess_design(source,design['canvas'],design,
        analyze(source,group_policies=record_data['request'].get('options',{}).get('group_policies'),
                overrides=record_data['request'].get('analysis_overrides') or None),
        'Human correction',record_data['request'].get('locks'))
    if not candidate['valid'] or not candidate['rubric']['gate'] or not candidate['quality']['passed']:
        raise HTTPException(422,{'message':'Edit violates preservation or readability requirements','checks':candidate['checks'],'rubric':candidate['rubric']})
    url,warnings=_render(design)
    if url is None:raise HTTPException(422,{'message':'Correction could not be verified by the official renderer','warnings':warnings})
    candidate.update(render=url,render_warnings=warnings,revision_hash=design_hash(design))
    candidate['verification']=_preview_verification(design,url)
    if candidate['revision_hash']==req.expected_revision_hash:raise HTTPException(422,'The edit did not change geometry')
    previous_controller=(current.get('candidate') or {}).get('controller')
    if previous_controller:
        candidate['controller']={**previous_controller,'incumbent_revision':candidate['revision_hash'],
            'active_revision':candidate['revision_hash'],'stop_reason':'human_correction',
            'events':previous_controller.get('events',[])+[{'type':'human_correction','operation':'Committed geometry edit',
                'revision_hash':candidate['revision_hash'],'verified':True}]}
    revision={"design":design,"candidate":candidate,"parent_revision":req.expected_revision_hash,
              "operation":"human_correction","accepted":True,"verified":True,"render":url,
              "resource_hash":candidate['verification']['resource_hash']}
    with LOCK:
        repo.put_revision(rid,target_index,revision)
        try:repo.compare_and_activate(rid,target_index,req.expected_revision_hash,candidate['revision_hash'])
        except Conflict as exc:raise HTTPException(409,str(exc))
        repo.feedback(rid,target_index,candidate['revision_hash'],'correction',other_hash=req.expected_revision_hash,reason=req.reason,
            context={'family':record_data['family'],'source_hash':design_hash(source),'target':design['canvas'],
                     'source_resource_hash':record_data.get('resource_hash'),'target_resource_hash':candidate['verification']['resource_hash'],
                     'options':record_data['request'].get('options',{})})
        record_data['response']['results'][target_index]['candidates'][0]=candidate
        save_run(record_data['request'],record_data['response'],rid)
    return {'candidate':candidate,'revision_hash':candidate['revision_hash']}


@router.post('/api/history/{rid}/targets/{target_index}/restore')
def restore_revision(rid:str,target_index:int,req:RestoreRevision):
    from service.app import _render,_preview_verification
    data=record(rid);repo=repository();revision=repo.revision(rid,target_index,req.revision_hash)
    current_job=repo.job(rid)
    if current_job and current_job['state'] in ('queued','running'):raise HTTPException(409,'Wait for this run to stop before restoring a revision')
    if not revision or not revision.get('candidate') or not revision.get('verified'):
        raise HTTPException(422,'Only a verified saved candidate can be restored')
    url,warnings=_render(revision['design'])
    if url is None:raise HTTPException(422,{'message':'Saved design no longer verifies with the current assets and fonts','warnings':warnings})
    candidate={**revision['candidate'],'render':url,'render_warnings':warnings,
               'verification':_preview_verification(revision['design'],url)}
    with LOCK:
        try:repo.compare_and_activate(rid,target_index,req.expected_revision_hash,req.revision_hash)
        except Conflict as exc:raise HTTPException(409,str(exc))
        revision.update(candidate=candidate,render=url,resource_hash=candidate['verification']['resource_hash'])
        repo.put_revision(rid,target_index,revision)
        data['response']['results'][target_index]['candidates'][0]=candidate
        save_run(data['request'],data['response'],rid)
        repo.append_event(rid,{'type':'restored','target_index':target_index,'revision_hash':req.revision_hash})
    return {'candidate':candidate,'revision_hash':req.revision_hash}
