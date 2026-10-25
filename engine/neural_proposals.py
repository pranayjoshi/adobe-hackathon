"""Isolated local visual arrangement proposals referencing existing group IDs.

Model output is data, never code. Every plan is validated with typed rejection
reasons *before* materialization, so containment repair cannot conceal a
malformed proposal (guide 15.2).

Plan schemas (guide 15.3):

* ``layout-plan/1`` (unchanged, geometry-only): ``{"groups": [{id,x,y,w,h}], "arrangement"?, "rationale"?}``.
  Normalised union box per layout block. Lossy: multi-member blocks are scaled
  uniformly and lone text is refitted, so a designer target may not round-trip.
* ``layout-plan/2``: ``{"schema": "layout-plan/2", "groups": [...], "elements": [{id,x,y,w,h,font_size?}], ...}``.
  Adds explicit per-element boxes (normalised to the target canvas) and integer
  target font sizes, so a finished design reconstructs exactly. Group boxes
  must agree with the union of their members.

Edit actions (``edit-action/1``) are typed geometry patches in target pixels
that become an ``engine.contracts.EditProposal`` and are materialized only by
``engine.materialize.apply_edit``.
"""
import copy,json,math,os,signal,subprocess,sys,tempfile,time
from dataclasses import dataclass,field
from pathlib import Path
from .canonical import ROOT,ASSETS_DIR,MUTABLE_FIELDS,backdrop_patch
from .quality import units
from .typography import fit_text

PLAN_V1='layout-plan/1'
PLAN_V2='layout-plan/2'
EDIT_V1='edit-action/1'
PLAN_SCHEMAS=(PLAN_V1,PLAN_V2)
GROUP_FIELDS=('id','x','y','w','h')
ELEMENT_FIELDS=('id','x','y','w','h','font_size','fit')  # fit: only the background 'cover' exception
TOP_FIELDS={PLAN_V1:{'groups','arrangement','rationale'},PLAN_V2:{'schema','groups','elements','arrangement','rationale'}}
BOUND_TOL=1e-3          # normalised slack for 4-decimal rounding of x+w / y+h
GROUP_TOL=.01           # v2: group box vs union of member boxes (normalised)
LOCK_TOL_PX=.5          # same tolerance as engine.materialize lock checks
ELEMENT_RANGE=(-1.0,2.0)  # v2 element origins may bleed (rotated stickers, cover backgrounds)
ELEMENT_MAX_SIZE=4.0
REASONS=('not_object','unknown_schema','unexpected_top_field','groups_not_list','elements_not_list','item_not_object',
         'missing_field','unexpected_field','id_not_string','unknown_id','duplicate_id','missing_block','missing_element',
         'boolean_value','non_numeric','non_finite','negative_origin','non_positive_size','out_of_bounds',
         'bad_font_size','bad_fit','bad_arrangement','bad_rationale','group_box_inconsistent','unknown_lock','partial_lock_v1','lock_conflict')


class PlanRejected(ValueError):
    """A plan failed typed validation; ``reasons`` lists every failure."""
    def __init__(self,reasons):
        self.reasons=tuple(reasons)
        super().__init__('invalid plan: '+', '.join(sorted({r['code'] for r in self.reasons})))


@dataclass(frozen=True)
class PlanCheck:
    ok:bool
    schema:str|None
    reasons:tuple=()

    @property
    def codes(self):return tuple(sorted({r['code'] for r in self.reasons}))

    def to_dict(self):return {'ok':self.ok,'schema':self.schema,'reasons':list(self.reasons)}


@dataclass
class PlanOutcome:
    """Typed result of plan -> design. ``design`` is None unless status == 'ok'."""
    status:str                  # ok | rejected | materialization_failed | invalid_design
    schema:str|None
    reasons:list=field(default_factory=list)
    design:dict|None=None
    raw_valid:bool=False        # passed validation + invariants without any containment repair
    repair_shift_px:float|None=None  # v1 only: how far containment moved anything
    raw_design:dict|None=None
    raw_checks:list=field(default_factory=list)
    repaired_valid:bool=False
    repaired_checks:list=field(default_factory=list)

    def to_dict(self):
        return {'status':self.status,'schema':self.schema,'reasons':self.reasons,'raw_valid':self.raw_valid,'repair_shift_px':self.repair_shift_px,'repaired_valid':self.repaired_valid,'raw_checks':self.raw_checks,'repaired_checks':self.repaired_checks}


def _infer_request(request, *, timeout=90, cancelled=None):
    """Bound the entire isolated render/model worker; no silent renderer fallback."""
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        return None,{'status':'timeout','model_calls':0,'renders':0}
    started=time.monotonic();event={'status':'model_failed','model_calls':0,'renders':0}
    with tempfile.TemporaryDirectory(prefix='chameleon-model-') as folder:
        req=Path(folder)/'request.json';answer=Path(folder)/'answer.json';request={**request,'timeout_seconds':timeout};req.write_text(json.dumps(request,allow_nan=False))
        proc=subprocess.Popen([sys.executable,'-m','engine.vlm_worker',str(req),str(answer)],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        reason=None
        while proc.poll() is None:
            if cancelled and cancelled():reason='cancelled';break
            if time.monotonic()-started>=timeout:reason='timeout';break
            time.sleep(min(.025,max(.001,timeout-(time.monotonic()-started))))
        if reason:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        if answer.is_file():
            try:event=json.loads(answer.read_text())
            except (ValueError,OSError):event={'status':'unparseable','model_calls':0,'renders':0}
        if reason:event={**event,'status':reason}
        event['seconds']=round(time.monotonic()-started,3)
        if event.get('status')!='generated':return None,event
        plan=parse_plan_text(event.get('raw')) if 'raw' in event else event.get('plan')
        if not isinstance(plan,dict):return None,{**event,'status':'unparseable'}
        return plan,{k:v for k,v in event.items() if k not in ('raw','plan')}


def propose(source,target,A,locks=None,*,timeout=90,cancelled=None):
    from .vlm import status
    st=status()
    if not st['enabled']:return None,{'status':'disabled','model_calls':0,'renders':0}
    if not st['weights_present']:return None,{'status':'weights_missing','detail':st.get('bundle_status'),'model_calls':0,'renders':0}
    if not st.get('runtime_available',True):return None,{'status':'runtime_unavailable','backend':st.get('backend'),'model_calls':0,'renders':0}
    schema=st.get('plan_schema') or PLAN_V1
    if schema not in PLAN_SCHEMAS:return None,{'status':'incompatible_plan_bundle','model_calls':0,'renders':0}
    prompt=plan_prompt(source,target,A) if schema==PLAN_V1 else plan_prompt_v2(source,target,A,locks)
    plan,event=_infer_request({'task':'plan','source':source,'target':target,'prompt':prompt,'schema':schema},timeout=timeout,cancelled=cancelled)
    if plan is None:return None,event
    outcome=materialize_plan(source,target,A,plan,locks)
    if outcome.status!='ok':return None,{**event,'status':outcome.status,'reasons':[r['code'] for r in outcome.reasons][:20]}
    return outcome.design,{**event,'status':'proposed','schema':outcome.schema,'raw_valid':outcome.raw_valid,'repaired_valid':outcome.repaired_valid}


def propose_edit(source,target,A,state,defects,locks=None,*,timeout=30,cancelled=None):
    """Return a revision-bound typed EditProposal, never a free-form replacement design."""
    from .contracts import DesignRevision
    from .vlm import status,check_bundle
    revision=state if isinstance(state,DesignRevision) else DesignRevision.create(state)
    st=status()
    if not st['enabled']:return None,{'status':'disabled','model_calls':0,'renders':0}
    if not st['weights_present']:return None,{'status':'weights_missing','model_calls':0,'renders':0}
    if not st.get('runtime_available'):return None,{'status':'runtime_unavailable','model_calls':0,'renders':0}
    if st.get('backend')=='transformers':
        bundle=check_bundle(os.environ.get('CHAMELEON_VLM_BUNDLE',''),verify_hashes=True)
        if bundle['status']!='ok' or 'edit' not in bundle.get('manifest',{}).get('tasks',[]):
            return None,{'status':'incompatible_edit_bundle','detail':bundle['status'],'model_calls':0,'renders':0}
    action,event=_infer_request({'task':'edit','source':source,'state':revision.design,'target':target,'schema':EDIT_V1,
                                  'prompt':edit_prompt(source,target,A,revision.design,defects,locks)},timeout=timeout,cancelled=cancelled)
    if action is None:return None,event
    try:proposal=action_to_proposal(action,revision,provenance='model',locks=locks)
    except (PlanRejected,ValueError,TypeError) as exc:
        return None,{**event,'status':'rejected','reasons':getattr(exc,'reasons',[{'code':'invalid_action'}])}
    return proposal,{**event,'status':'proposed','schema':EDIT_V1,'base_revision':revision.revision_hash}


def parse_plan_text(text):
    """First JSON object in a model reply, or None. Never evaluates anything."""
    if not isinstance(text,str):return None
    text=text.strip()
    if text.startswith('```'):text=text.strip('`').split('\n',1)[-1]
    try:
        start,end=text.index('{'),text.rindex('}')+1
        value=json.loads(text[start:end])
    except ValueError:return None
    return value if isinstance(value,dict) else None


# ---------------------------------------------------------------- validation

def _num(value):
    """Reason code for a value that is not a usable finite number, else None."""
    if isinstance(value,bool):return 'boolean_value'
    if not isinstance(value,(int,float)):return 'non_numeric'
    if not math.isfinite(value):return 'non_finite'
    return None


def plan_schema(plan):
    if not isinstance(plan,dict):return None
    s=plan.get('schema',PLAN_V1)
    return s if s in PLAN_SCHEMAS else None


def _check_items(items,fields,required,reasons,where):
    """Field/number checks shared by group and element entries.

    Returns (well_formed, identified): entries usable for geometry checks, and every
    entry with a string id (used for coverage, so one bad value is not also 'missing')."""
    good=[];named=[]
    for n,p in enumerate(items):
        if not isinstance(p,dict):reasons.append({'code':'item_not_object','where':where,'index':n});continue
        bad=False
        for k in sorted(set(p)-set(fields),key=str):reasons.append({'code':'unexpected_field','where':where,'index':n,'field':str(k)[:40]});bad=True
        for k in required:
            if k not in p:reasons.append({'code':'missing_field','where':where,'index':n,'field':k});bad=True
        if 'id' in p and not isinstance(p['id'],str):reasons.append({'code':'id_not_string','where':where,'index':n});bad=True
        elif 'id' in p:named.append(p)
        for k in ('x','y','w','h','font_size'):
            if k in p:
                code=_num(p[k])
                if code:reasons.append({'code':code,'where':where,'id':p.get('id') if isinstance(p.get('id'),str) else None,'field':k});bad=True
        if not bad:good.append(p)
    return good,named


def _coverage(items,expected,reasons,where,missing_code,required=True):
    ids=[p['id'] for p in items]
    seen=set()
    for i in ids:
        if i in seen:reasons.append({'code':'duplicate_id','where':where,'id':i})
        seen.add(i)
        if i not in expected[1]:reasons.append({'code':'unknown_id','where':where,'id':i})
    if required:
        for i in sorted(set(expected[0])-seen):reasons.append({'code':missing_code,'where':where,'id':i})


def validate_plan(A,plan,*,locks=None,target=None,source=None):
    """Typed plan validation. ``plan`` is a v1/v2 dict or a legacy list of v1 groups.

    Exact block coverage, no duplicate/unknown/non-string IDs, finite non-boolean
    numbers, non-negative origins, positive sizes, sums within the unit canvas,
    allowlisted fields, and lock compatibility (``locks`` = {element_id: target-pixel
    rect}, requires ``target``). Nothing is clamped or repaired here.
    """
    reasons=[]
    if isinstance(plan,list):plan={'groups':plan}
    if not isinstance(plan,dict):return PlanCheck(False,None,({'code':'not_object'},))
    schema=plan_schema(plan)
    if schema is None:return PlanCheck(False,None,({'code':'unknown_schema','value':str(plan.get('schema'))[:40]},))
    for k in sorted(set(plan)-TOP_FIELDS[schema],key=str):reasons.append({'code':'unexpected_top_field','field':str(k)[:40]})
    if 'arrangement' in plan:
        from .constitution import ARRANGEMENTS
        if plan['arrangement'] not in ARRANGEMENTS:reasons.append({'code':'bad_arrangement'})
    if 'rationale' in plan and not isinstance(plan['rationale'],str):reasons.append({'code':'bad_rationale'})
    groups=plan.get('groups')
    if not isinstance(groups,list):return PlanCheck(False,schema,tuple(reasons+[{'code':'groups_not_list'}]))
    blocks={b.id:b for b in A.all_blocks()}
    good,named=_check_items(groups,GROUP_FIELDS,GROUP_FIELDS,reasons,'groups')
    # Entries with non-string ids never cover a block: mixed invalid IDs cannot pass.
    _coverage(named,(blocks,blocks),reasons,'groups','missing_block')
    for p in good:
        i=p['id']
        if p['x']<0 or p['y']<0:reasons.append({'code':'negative_origin','where':'groups','id':i})
        if p['w']<=0 or p['h']<=0:reasons.append({'code':'non_positive_size','where':'groups','id':i})
        if p['x']>1 or p['y']>1 or p['x']+p['w']>1+BOUND_TOL or p['y']+p['h']>1+BOUND_TOL:reasons.append({'code':'out_of_bounds','where':'groups','id':i})
    elements={}
    if schema==PLAN_V2:
        items=plan.get('elements')
        if not isinstance(items,list):reasons.append({'code':'elements_not_list'});items=[]
        member_ids={e['id'] for b in blocks.values() for e in b.members}
        all_ids=set(source_ids(A,source)) | member_ids
        texts={e['id'] for b in blocks.values() for e in b.members if e['type']=='text'}
        if source:texts|={e['id'] for e in source['elements'] if e['type']=='text'}
        egood,enamed=_check_items(items,ELEMENT_FIELDS,GROUP_FIELDS,reasons,'elements')
        _coverage(enamed,(all_ids,all_ids),reasons,'elements','missing_element')
        lo,hi=ELEMENT_RANGE
        for p in egood:
            i=p['id']
            if p['w']<=0 or p['h']<=0:reasons.append({'code':'non_positive_size','where':'elements','id':i})
            elif p['w']>ELEMENT_MAX_SIZE or p['h']>ELEMENT_MAX_SIZE:reasons.append({'code':'out_of_bounds','where':'elements','id':i})
            if not (lo<=p['x']<=hi and lo<=p['y']<=hi):reasons.append({'code':'out_of_bounds','where':'elements','id':i})
            if 'fit' in p and not (p['fit']=='cover' and source is not None and _cover_ok(source,i)):reasons.append({'code':'bad_fit','id':i})
            if 'font_size' in p:
                fs=p['font_size']
                if i not in texts or isinstance(fs,float) and not fs.is_integer() or fs<1:reasons.append({'code':'bad_font_size','id':i})
            elements[i]=p
        # Group boxes must describe their members; otherwise the plan is self-contradictory.
        # Rotation is applied in target pixels (normalised space is anisotropic).
        from .canonical import union_all,rotated_extent
        W,H=(float(target['width']),float(target['height'])) if target else (1.0,1.0)
        for p in good:
            b=blocks.get(p['id'])
            if b is None or not all(e['id'] in elements for e in b.members):continue
            rot={e['id']:e.get('rotation',0) or 0 for e in b.members}
            if any(rot.values()) and not target:continue
            u=union_all(rotated_extent({'x':elements[i]['x']*W,'y':elements[i]['y']*H,'width':elements[i]['w']*W,'height':elements[i]['h']*H,'rotation':rot[i]}) for i in b.ids).clip(W,H)
            if max(abs(u.x/W-p['x']),abs(u.y/H-p['y']),abs(u.w/W-p['w']),abs(u.h/H-p['h']))>GROUP_TOL:
                reasons.append({'code':'group_box_inconsistent','id':p['id']})
    if locks:
        reasons+=_lock_reasons(A,schema,good,elements,locks,target)
    return PlanCheck(not reasons,schema,tuple(reasons))


def _cover_ok(source,eid):
    from .scene import background_cover_allowed
    e=next((x for x in source['elements'] if x['id']==eid),None)
    return e is not None and background_cover_allowed(source,e,{**e,'fit':'cover'})


def source_ids(A,source=None):
    ids=[e['id'] for b in A.all_blocks() for e in b.members]+[e['id'] for e in A.background]+[e['id'] for e in A.decorations]
    if source:ids+=[e['id'] for e in source['elements']]
    return ids


def _lock_reasons(A,schema,groups,elements,locks,target):
    out=[]
    if not target:return [{'code':'lock_conflict','detail':'locks require a target canvas'}]
    W,H=float(target['width']),float(target['height'])
    member_of={e['id']:b for b in A.all_blocks() for e in b.members}
    known=set(source_ids(A))
    gbox={p['id']:p for p in groups}
    for eid,rect in sorted(locks.items()):
        if eid not in known:out.append({'code':'unknown_lock','id':eid});continue
        b=member_of.get(eid)
        if schema==PLAN_V2 and eid in elements:
            p=elements[eid];box={'x':p['x']*W,'y':p['y']*H,'width':p['w']*W,'height':p['h']*H}
            if any(abs(box[k]-v)>LOCK_TOL_PX for k,v in rect.items() if k in box):out.append({'code':'lock_conflict','id':eid})
            continue
        if b is None:continue  # Legacy plans derive these elements deterministically.
        if not all(m['id'] in locks for m in b.members):out.append({'code':'partial_lock_v1','id':eid,'block':b.id});continue
        p=gbox.get(b.id)
        if p is None:continue
        from .canonical import union_all,Box
        u=union_all(Box(locks[m['id']].get('x',0),locks[m['id']].get('y',0),locks[m['id']].get('width',0),locks[m['id']].get('height',0)) for m in b.members)
        if max(abs(u.x-p['x']*W),abs(u.y-p['y']*H),abs(u.w-p['w']*W),abs(u.h-p['h']*H))>LOCK_TOL_PX+BOUND_TOL*max(W,H):
            out.append({'code':'lock_conflict','id':eid,'block':b.id})
    return out


def plan_valid(A,placements):
    """Compatibility predicate (training.reward, training.train_vlm): strict typed validation."""
    return validate_plan(A,placements).ok


# ------------------------------------------------------------ materialization

def apply_plan(source,target,A,placements,*,locks=None,validate=True):
    """Turn a plan into a design. ``placements`` is a legacy v1 group list or a v1/v2 plan dict.

    Invalid plans raise PlanRejected (a ValueError) before any containment repair.
    v1: uniform per-block scaling, measured text, containment (unchanged behaviour).
    v2: exact per-element geometry and font sizes; no containment (raw output).
    """
    plan={'groups':placements} if isinstance(placements,list) else placements
    if validate:
        check=validate_plan(A,plan,locks=locks,target=target,source=source)
        if not check.ok:raise PlanRejected(check.reasons)
    schema=plan_schema(plan) or PLAN_V1
    out=_baseline(source,target,A)
    if schema==PLAN_V2:
        out=_apply_v2(out,target,plan)
    else:
        out=_apply_v1(out,target,A,plan['groups'])
        from .repair import contain
        out=contain(out,A)
    return _pin_locks(out,locks)


def _baseline(source,target,A):
    out={'canvas':{'width':int(target['width']),'height':int(target['height'])},'elements':copy.deepcopy(source['elements'])};by={e['id']:e for e in out['elements']}
    sx,sy=target['width']/A.sw,target['height']/A.sh
    for e in out['elements']:
        e.update(x=e['x']*sx,y=e['y']*sy,width=e['width']*min(sx,sy),height=e['height']*min(sx,sy))
        if e['type']=='text':e['font_size']=max(1,round(e['font_size']*min(sx,sy)))
    for bg in A.background:by[bg['id']].update(backdrop_patch(bg,target['width'],target['height']))
    return out


def _apply_v1(out,target,A,placements):
    by={e['id']:e for e in out['elements']};boxes={p['id']:p for p in placements}
    src={e['id']:e for b in A.all_blocks() for e in b.members}
    for b in A.all_blocks():
        p=boxes[b.id];r=b.src_box;kw,kh=p['w']*target['width']/max(1,r.w),p['h']*target['height']/max(1,r.h);k=min(kw,kh)
        if len(b.members)==1 and b.members[0]['type']=='text' and not b.members[0].get('rotation'):
            # A lone text block reflows: the box is the plan, the font is the largest that fits it.
            e=b.members[0];o=by[e['id']];w,h=p['w']*target['width'],p['h']*target['height']
            o.update(x=p['x']*target['width'],y=p['y']*target['height'],width=w,height=h)
            o['font_size']=fit_text(o,w,h,max(1,round(e['font_size']*min(4,max(kw,kh))))).font_size
            continue
        for e in b.members:
            e=src[e['id']];o=by[e['id']];o.update(x=p['x']*target['width']+(e['x']-r.x)*k,y=p['y']*target['height']+(e['y']-r.y)*k,width=e['width']*k,height=e['height']*k)
            if o['type']=='text':o['font_size']=fit_text(o,o['width'],o['height'],max(1,round(e['font_size']*k))).font_size
    return out


def _apply_v2(out,target,plan):
    by={e['id']:e for e in out['elements']};W,H=target['width'],target['height']
    for p in plan['elements']:
        o=by[p['id']];o.update(x=p['x']*W,y=p['y']*H,width=p['w']*W,height=p['h']*H)
        if 'fit' in p:o['fit']=p['fit']
        if 'font_size' in p:o['font_size']=int(p['font_size'])
        elif o['type']=='text':o['font_size']=fit_text(o,o['width'],o['height'],int(o['font_size'])).font_size
    return out


def _pin_locks(out,locks):
    if locks:
        by={e['id']:e for e in out['elements']}
        for eid,rect in locks.items():
            if eid in by:by[eid].update({k:v for k,v in rect.items() if k in MUTABLE_FIELDS})
    return out


def materialize_plan(source,target,A,plan,locks=None):
    """Measure raw and containment-repaired invariants independently."""
    from .invariants import passes,validate
    check=validate_plan(A,plan,locks=locks,target=target,source=source)
    if not check.ok:return PlanOutcome('rejected',check.schema,list(check.reasons))
    try:
        raw=_baseline(source,target,A)
        raw=_apply_v2(raw,target,plan) if check.schema==PLAN_V2 else _apply_v1(raw,target,A,plan if isinstance(plan,list) else plan['groups'])
        raw=_pin_locks(raw,locks);raw_checks=validate(source,raw)
        design=apply_plan(source,target,A,plan,locks=locks,validate=False);checks=validate(source,design)
    except (KeyError,ValueError,TypeError,ZeroDivisionError) as exc:
        return PlanOutcome('materialization_failed',check.schema,[{'code':'materialization_failed','detail':type(exc).__name__}])
    raw_by={e['id']:e for e in raw['elements']}
    shift=round(max([abs(raw_by[e['id']][k]-e[k]) for e in design['elements'] for k in ('x','y','width','height')] or [0]),3)
    repaired_valid=passes(checks);raw_valid=passes(raw_checks)
    reasons=[] if repaired_valid else [{'code':'invariant','name':c['name']} for c in checks if not c['passed'] and c['severity']=='error']
    return PlanOutcome('ok' if repaired_valid else 'invalid_design',check.schema,reasons,design if repaired_valid else None,
                       raw_valid,shift,raw,raw_checks,repaired_valid,checks)


def plan_fidelity(reference,materialized,exact_px=1.0,near_px=2.0,near_frac=.005):
    """Reconstruction error of a materialized plan against its intended design.

    ``class``: exact (<= exact_px, identical font sizes) | near (<= max(near_px, near_frac*canvas),
    font sizes within 1) | lossy. Lossy targets must not become SFT completions.
    """
    ref={e['id']:e for e in reference['elements']};got={e['id']:e for e in materialized['elements']}
    if set(ref)!=set(got) or reference['canvas']!=materialized['canvas']:return {'class':'lossy','max_px':None,'font_max':None,'worst_id':None}
    worst=(0.0,None);font=0
    for i,e in ref.items():
        d=max(abs(float(got[i][k])-float(e[k])) for k in ('x','y','width','height'))
        if d>worst[0]:worst=(d,i)
        if e['type']=='text':font=max(font,abs(int(got[i]['font_size'])-int(e['font_size'])))
        if e.get('fit')!=got[i].get('fit'):return {'class':'lossy','max_px':None,'font_max':font,'worst_id':i,'fit_mismatch':True}
    W,H=reference['canvas']['width'],reference['canvas']['height']
    cls='exact' if worst[0]<=exact_px and font==0 else 'near' if worst[0]<=max(near_px,near_frac*max(W,H)) and font<=1 else 'lossy'
    return {'class':cls,'max_px':round(worst[0],3),'font_max':font,'worst_id':worst[1]}


# ---------------------------------------------------------------- prompts/IO

def block_table(A):
    """What a model sees per block. Copy length only, never the copy, so it cannot learn to rewrite."""
    from .typography import source_lines
    rows=[]
    for b in A.all_blocks():
        r=b.src_box;texts=[e for e in b.members if e['type']=='text']
        rows.append({'id':b.id,'role':b.role,'kind':type(b).__name__.replace('Block','').lower(),'members':len(b.members),
                     'box':[round(r.x/A.sw,3),round(r.y/A.sh,3),round(r.w/A.sw,3),round(r.h/A.sh,3)],
                     'font_size':max((e['font_size'] for e in texts),default=None),'chars':sum(len(str(e.get('content',''))) for e in texts),
                     'lines':sum(source_lines(e) for e in texts)})
    return rows


def element_table(A,design=None):
    """Per-element geometry for v2 plans and edits; ids, types and boxes only, never copy or assets."""
    by={e['id']:e for e in design['elements']} if design else None
    cw,ch=(design['canvas']['width'],design['canvas']['height']) if design else (A.sw,A.sh)
    rows=[]
    for b in A.all_blocks():
        for m in b.members:
            e=by[m['id']] if by else m
            row={'id':e['id'],'block':b.id,'type':e['type'],'box':[round(e['x']/cw,4),round(e['y']/ch,4),round(e['width']/cw,4),round(e['height']/ch,4)]}
            if e.get('rotation'):row['rotation']=round(float(e['rotation']),1)
            if e['type']=='text':row['font_size']=int(e['font_size'])
            rows.append(row)
    return rows


def plan_prompt(source,target,A):
    from .constitution import CONSTITUTION
    from .formats import name_for
    return (CONSTITUTION+f"\n\nSource canvas: {A.sw:g}x{A.sh:g}. Target: \"{name_for(target['width'],target['height'])}\" {target['width']}x{target['height']}."
            "\nReturn every block exactly once as normalized boxes (0-1) with positive sizes inside the canvas, plus the arrangement you chose."
            "\nBlocks (box = source x, y, w, h normalized): "+json.dumps(block_table(A)))


def plan_prompt_v2(source,target,A,locks=None):
    from .constitution import CONSTITUTION
    from .formats import name_for
    lk=sorted((locks or {}))
    return (CONSTITUTION+f"\n\nSource canvas: {A.sw:g}x{A.sh:g}; aspect {A.sw/A.sh:.3f}. Target: \"{name_for(target['width'],target['height'])}\" {target['width']}x{target['height']}."
            f"\nReply with JSON only: {{\"schema\": \"{PLAN_V2}\", \"arrangement\": ..., \"groups\": [{{id,x,y,w,h}}], \"elements\": [{{id,x,y,w,h,font_size}}]}}."
            "\nEvery block and every listed element exactly once; boxes normalized to the target canvas; font_size is an integer in target pixels (text only); group boxes equal the union of their elements."
            +(f"\nLocked elements (keep exactly): {json.dumps(lk)}" if lk else '')
            +"\nBlocks (box = source x, y, w, h normalized): "+json.dumps(block_table(A))
            +"\nElements (box = source x, y, w, h normalized): "+json.dumps(element_table(A)))


def edit_prompt(source,target,A,state,defects,locks=None):
    """Prompt for one typed edit on the current state (edit-imitation policy)."""
    from .constitution import CONSTITUTION
    lk=sorted((locks or {}))
    return (CONSTITUTION+f"\n\nTarget canvas {target['width']}x{target['height']}. Propose ONE geometry edit that fixes the listed defects or improves the layout."
            f"\nReply with JSON only: {{\"schema\": \"{EDIT_V1}\", \"operation\": str, \"patch\": {{element_id: {{x,y,width,height,font_size}}}}}} in target pixels; only listed ids; never other fields."
            +(f"\nLocked elements (never edit): {json.dumps(lk)}" if lk else '')
            +"\nCurrent defects: "+json.dumps(defects,sort_keys=True)
            +"\nBlocks: "+json.dumps(block_table(A))
            +"\nCurrent elements (box normalized to target): "+json.dumps(element_table(A,state)))


def design_to_plan(design,A,arrangement=None,schema=PLAN_V1):
    """Inverse of apply_plan. v1: normalised union box per block. v2 adds exact element boxes/fonts."""
    from .canonical import union_all,rotated_extent
    by={e['id']:e for e in design['elements']};cw,ch=design['canvas']['width'],design['canvas']['height'];groups=[]
    for b in A.all_blocks():
        r=union_all(rotated_extent(by[i]) for i in b.ids).clip(cw,ch)
        groups.append({'id':b.id,'x':round(r.x/cw,4),'y':round(r.y/ch,4),'w':round(max(1e-4,r.w/cw),4),'h':round(max(1e-4,r.h/ch),4)})
    plan={'groups':groups}
    if arrangement:plan['arrangement']=arrangement
    if schema==PLAN_V2:
        elements=[]
        for e in design['elements']:
            row={'id':e['id'],'x':round(e['x']/cw,4),'y':round(e['y']/ch,4),'w':round(max(1e-4,e['width']/cw),4),'h':round(max(1e-4,e['height']/ch),4)}
            if e['type']=='text':row['font_size']=int(e['font_size'])
            elements.append(row)
        plan={'schema':PLAN_V2,**plan,'elements':elements}
    return plan


def design_to_plan_v2(design,source,A,arrangement=None):
    """Exact v2 geometry for every source element, including backgrounds and decorations."""
    plan=design_to_plan(design,A,arrangement,PLAN_V2)
    srcfit={e['id']:e.get('fit') for e in source['elements']}
    for row in plan['elements']:
        e=next(x for x in design['elements'] if x['id']==row['id'])
        if e.get('fit')!=srcfit[row['id']]:row['fit']=e.get('fit')
    base={e['id']:e for e in _baseline(source,design['canvas'],A)['elements']}
    listed={p['id'] for p in plan['elements']};cw,ch=design['canvas']['width'],design['canvas']['height']
    for e in design['elements']:
        if e['id'] in listed:continue
        if max(abs(base[e['id']][k]-e[k]) for k in ('x','y','width','height'))>LOCK_TOL_PX or (e['type']=='text' and base[e['id']]['font_size']!=e['font_size']) or e.get('fit')!=base[e['id']].get('fit'):
            row={'id':e['id'],'x':round(e['x']/cw,4),'y':round(e['y']/ch,4),'w':round(max(1e-4,e['width']/cw),4),'h':round(max(1e-4,e['height']/ch),4)}
            if e['type']=='text':row['font_size']=int(e['font_size'])
            if e.get('fit')!=base[e['id']].get('fit'):row['fit']=e.get('fit')
            plan['elements'].append(row)
    return plan


# -------------------------------------------------------------- edit actions

def edit_action(design_before,design_after,operation):
    """Canonical edit-action/1 JSON from two designs on the same canvas (values pre-rounded
    exactly as engine.materialize.apply_edit stores them)."""
    old={e['id']:e for e in design_before['elements']};patch={}
    for e in design_after['elements']:
        delta={}
        for k in sorted(MUTABLE_FIELDS):
            if k in e and old[e['id']].get(k)!=e[k]:
                v=max(1,int(e[k])) if k=='font_size' else round(float(e[k]),2)
                if (max(1,int(old[e['id']][k])) if k=='font_size' else round(float(old[e['id']][k]),2))!=v:delta[k]=v
        if e.get('fit')!=old[e['id']].get('fit'):delta['fit']=e.get('fit')
        if delta:patch[e['id']]=delta
    return {'schema':EDIT_V1,'operation':str(operation),'patch':dict(sorted(patch.items()))}


def validate_edit_action(action,state,locks=None):
    """Typed checks before an action becomes an EditProposal; apply_edit re-checks everything."""
    reasons=[]
    if not isinstance(action,dict):return [{'code':'not_object'}]
    if action.get('schema',EDIT_V1)!=EDIT_V1:reasons.append({'code':'unknown_schema'})
    for k in sorted(set(action)-{'schema','operation','patch','affected_ids','intended_improvement'},key=str):reasons.append({'code':'unexpected_top_field','field':str(k)[:40]})
    if not isinstance(action.get('operation'),str):reasons.append({'code':'missing_field','field':'operation'})
    patch=action.get('patch')
    if not isinstance(patch,dict) or not patch:return reasons+[{'code':'missing_field','field':'patch'}]
    by={e['id']:e for e in state['elements']}
    for eid,p in patch.items():
        if eid not in by:reasons.append({'code':'unknown_id','id':str(eid)[:40]});continue
        if locks and eid in locks:reasons.append({'code':'lock_conflict','id':eid})
        if not isinstance(p,dict) or not p:reasons.append({'code':'item_not_object','id':eid});continue
        for k,v in p.items():
            if k=='fit':
                if v!='cover':reasons.append({'code':'bad_fit','id':eid})
                continue  # apply_edit admits it only for an eligible canvas background
            if k not in MUTABLE_FIELDS:reasons.append({'code':'unexpected_field','id':eid,'field':str(k)[:40]});continue
            code=_num(v)
            if code:reasons.append({'code':code,'id':eid,'field':k});continue
            if k in ('width','height') and v<=0:reasons.append({'code':'non_positive_size','id':eid,'field':k})
            if k=='font_size' and (v<1 or (by[eid]['type']!='text')):reasons.append({'code':'bad_font_size','id':eid})
    return reasons


def action_to_proposal(action,revision,provenance='model',locks=None):
    """edit-action/1 -> engine.contracts.EditProposal against ``revision`` (raises PlanRejected)."""
    from .contracts import EditProposal
    reasons=validate_edit_action(action,revision.design,locks)
    if reasons:raise PlanRejected(reasons)
    patch={k:dict(v) for k,v in sorted(action['patch'].items())}
    return EditProposal(revision.revision_hash,action['operation'],patch,tuple(patch),action.get('intended_improvement',''),provenance)


def arrangement_of(template):
    t=template.replace('solver_','')
    for prefix,name in (('stack','stack'),('split','side_by_side'),('columns','side_by_side'),('banner','banner_row'),('anchored','anchored')):
        if t.startswith(prefix):return name
    return 'source_like' if t.split('_scale')[0] in ('scale','source_frame','identity','photo_scene') else None
