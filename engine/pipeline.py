"""End-to-end pipeline: analyse -> propose -> solve -> serialise -> validate -> score.

`generate()` returns ranked, validated candidates for the viewer/service;
`resize()` in resize.py returns the top one for the competition runner.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from .analysis import Analysis, analyze
from .canonical import element_box
from .invariants import passes, validate
from .layout import Placement, propose
from .metrics import score


def _round(v: float) -> float:
    return round(float(v), 2)


def serialise(source: dict[str, Any], target: dict[str, Any], P: Placement) -> dict[str, Any]:
    """Patch allowlisted geometry onto an untouched copy of every source element."""
    cw, ch = int(target["width"]), int(target["height"])
    sx, sy = cw / source["canvas"]["width"], ch / source["canvas"]["height"]
    s = min(sx, sy)
    elements = []
    for el in source["elements"]:
        out = copy.deepcopy(el)
        patch = P.patches.get(el["id"])
        if patch is None:  # defensive: anything unplaced is scaled proportionally
            b = element_box(el)
            patch = {"x": b.x * sx, "y": b.y * sy, "width": b.w * s, "height": b.h * s}
            if el["type"] == "text":
                patch["font_size"] = max(1, round(float(el["font_size"]) * s))
        w, h = max(0.0, patch["width"]), max(0.0, patch["height"])
        if w > cw or h > ch:  # last-resort containment, aspect preserved
            k = min(cw / w if w else 1, ch / h if h else 1)
            w, h = w * k, h * k
        x = min(max(0.0, patch["x"]), cw - w)
        y = min(max(0.0, patch["y"]), ch - h)
        out.update({"x": _round(x), "y": _round(y), "width": _round(w), "height": _round(h)})
        if el["type"] == "text" and "font_size" in patch:
            out["font_size"] = max(1, int(patch["font_size"]))
        elements.append(out)
    return {"canvas": {"width": cw, "height": ch}, "elements": elements}


def uniform_stretch(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """The naive baseline from the starter kit, kept for comparison."""
    sx = target["width"] / source["canvas"]["width"]
    sy = target["height"] / source["canvas"]["height"]
    els = []
    for el in source["elements"]:
        el = dict(el)
        el.update(x=el["x"] * sx, y=el["y"] * sy, width=el["width"] * sx, height=el["height"] * sy)
        if el["type"] == "text":
            el["font_size"] = max(1.0, el["font_size"] * min(sx, sy))
        els.append(el)
    return {"canvas": {"width": target["width"], "height": target["height"]}, "elements": els}


def _placement(A, design, template="candidate"):
    from .quality import units
    from .canonical import union_all
    p=Placement(template,template,"Measured candidate")
    for block,rect in units(A,design):
        p.boxes[block.id]=rect
        p.overlay.append({"id":block.id,**rect.to_dict()})
    return p


def _lock(design,A,locks):
    from .quality import units
    by={e["id"]:e for e in design["elements"]}
    for block,rect in units(A,design):
        chosen=next((i for i in block.ids if i in locks),None)
        if chosen is None:continue
        e=by[chosen];r=locks[chosen]
        k=min(r.get("width",e["width"])/max(.001,e["width"]),r.get("height",e["height"])/max(.001,e["height"]))
        x,y=e["x"],e["y"]
        for eid in block.ids:
            o=by[eid];o.update(x=r["x"]+(o["x"]-x)*k,y=r["y"]+(o["y"]-y)*k,width=o["width"]*k,height=o["height"]*k)
            if o["type"]=="text":o["font_size"]=max(1,int(o["font_size"]*k))
    return design


def _provider_workflow(options):
    import os
    return (options.get('proposal_portfolio',os.environ.get('CHAMELEON_PROPOSAL_PORTFOLIO','legacy'))=='diverse'
            or options.get('retrieval_version',os.environ.get('CHAMELEON_RETRIEVAL_VERSION','legacy'))=='fit_aware'
            or options.get('photo_mode',os.environ.get('CHAMELEON_PHOTO_ANALYSIS','heuristic'))=='subject_aware'
            or options.get('background_mode')=='safe_cover')


def _generate_legacy(source, target, emphasis="balanced", top_k=3, analysis=None, *, locks=None,
             options=None, all_candidates=False):
    """Search -> repair -> diagnose -> render shortlist -> rank -> validate.

    Optional learned components add proposals; all outputs pass the same checks.
    """
    import hashlib,json,time
    from .judging import judge_quality
    from .quality import diagnose
    from .repair import repair
    from .constraints import solve
    from .learning import apply_roles,preference_score
    from .retrieval import retrieve
    from .neural_proposals import propose as neural
    from .rubric import evaluate as rubric_evaluate
    started=time.monotonic();locks=locks or {};options=options or {}
    A=copy.deepcopy(analysis) if analysis else analyze(source,group_policies=options.get("group_policies"))
    role_status=apply_roles(source,A,enabled=options.get("role_mode",options.get("roles")))
    if _provider_workflow(options):
        from .providers import ProposalContext, legacy_proposals
        context=ProposalContext(copy.deepcopy(source),dict(target),A,locks,dict(options),emphasis,role_status=role_status)
        proposals,events,provider_results=legacy_proposals(context,
            portfolio=options.get('proposal_portfolio'),retrieval_version=options.get('retrieval_version'))
        events.append({'providers':[r.to_dict() for r in provider_results]})
    else:
        proposals=[];events=[]
        for P in propose(A,target["width"],target["height"],emphasis):
            proposals.append((P.template,P.name,P.description,serialise(source,target,P)))
        if options.get("retrieval",True):
            try:
                proposals += [(f"retrieved_{i}","Retrieved arrangement","Matched to training geometry",d) for i,d in enumerate(retrieve(source,target,A))]
            except (ValueError,OSError,KeyError):events.append({"retrieval":"invalid library ignored"})
        if options.get("vlm",False):
            d,event=neural(source,target,A);events.append({"vlm":event})
            if d:proposals.append(("visual_proposal","Visual proposal","Suggested by the configured local model",d))
        from .scene import propose_photo_scene
        proposals += [('photo_scene','Photo with copy column','Original background cropped without distortion; copy stays in its original horizontal region.',d) for d in propose_photo_scene(source,target,A)]
        # Explicit identity candidate prevents unnecessary redesign on same-size tasks.
        if source["canvas"]==target:proposals.insert(0,("identity","Original composition","Already at the requested size",copy.deepcopy(source)))
        # Preserve the original geometry as a real safety candidate. It competes
        # through the same quality gates; the distorted stretch reference does not.
        safe=copy.deepcopy(source);safe['canvas']=dict(target)
        from .canonical import union_all,rotated_extent
        frame=union_all([rotated_extent(e) for e in source['elements']]+[element_box({'x':0,'y':0,'width':A.sw,'height':A.sh})])
        k=min(target['width']/frame.w,target['height']/frame.h)
        ox=(target['width']-frame.w*k)/2-frame.x*k;oy=(target['height']-frame.h*k)/2-frame.y*k
        for e in safe['elements']:
            e.update(x=e['x']*k+ox,y=e['y']*k+oy,width=e['width']*k,height=e['height']*k)
            if e['type']=='text':e['font_size']=max(1,int(e['font_size']*k))
        proposals.append(('source_frame','Original arrangement','Uniform scaling preserves the original arrangement; may leave unused space.',safe))
        # Layout transformer trained on Crello designer resizes (engine/retarget); its boxes pass
        # the same checks as every other candidate.
        from .retarget.propose import candidates as learned
        proposals+=learned(source,target,A,options,events)
    results=[];seen=set()
    def build(template,name,description,design,trace):
        # Dataset provenance belongs to the source record, not the strict
        # competition design schema. Every candidate exports the same envelope.
        design={"canvas":{"width":int(target["width"]),"height":int(target["height"])},"elements":design["elements"]}
        design=_lock(design,A,locks)
        key=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()[:16]
        if key in seen:return None
        seen.add(key)
        p=_placement(A,design,template);checks=validate(source,design)
        for eid,rect in locks.items():
            el=next((e for e in design["elements"] if e["id"]==eid),{})
            ok=all(abs(el.get(k,-1e9)-v)<.6 for k,v in rect.items() if k in ("x","y","width","height"))
            checks.append({"name":"Locked placement: "+eid,"passed":ok,"detail":"Exact placement retained" if ok else "Cannot satisfy this lock","severity":"error"})
        q=score(source,design,A,p);diagnostics=diagnose(source,design,A)
        return {"id":key,"template":template,"name":name,"description":description,"design":design,"checks":checks,"valid":passes(checks),"score":q,"overlay":p.overlay,"diagnostics":diagnostics,"trace":trace,"quality":judge_quality(source,design,A,diagnostics)}
    def evaluate(*args):
        r=build(*args)
        if r:results.append(r)
        return r
    def finish(r):
        """Quality gates, penalties and rubric. Idempotent: safe to rerun after new diagnostics."""
        r['quality']=judge_quality(source,r['design'],A,r['diagnostics'])
        d=r["diagnostics"];base=r["score"].setdefault("base",{"total":r["score"]["total"],"issues":list(r["score"]["issues"])})
        issues=r["score"]["issues"]=list(base["issues"])
        for key,message in [('unsupported_text','Text escapes its original photo background'),('scene_displacement','Text has moved away from its original area of the photograph')]:
            if r['quality']['defects'].get(key):issues.append({'severity':'error','message':message,'ids':r['quality']['defects'][key]})
        if r['quality']['defects']['unreadable']:
            issues.append({'severity':'error','message':'Text is below the readable minimum for this format','ids':r['quality']['defects']['unreadable']})
        for kind,ids,message in (("overflow",r["quality"]["defects"]["overflow"],"Text exceeds its measured box"),("contrast",d["contrast_regressions"],"Text loses contrast against its background")):
            if ids:issues.append({"severity":"error","message":message,"ids":ids})
        if d["collisions"]:issues.append({"severity":"error","message":"New or worsened group overlaps","ids":list({i for c in d["collisions"] for i in c["ids"]})})
        # Show an explicit quality penalty; compliance remains independently visible.
        r["score"]["total"]=round(max(0,base["total"]-8*len(r["quality"]["defects"]["overflow"])-6*len(d["collisions"])-5*len(d["contrast_regressions"])),1)
        if not r['quality']['passed']:
            r['score']['total']=min(r['score']['total'],max(0,45-3*r['quality']['defect_count']))
        r["rubric"]=rubric_evaluate(source,r["design"],A,r["diagnostics"],r["score"],target)
        return r
    for template,name,desc,design in proposals:
        if template in ("identity","source_frame"):evaluate(template,name,desc,design,[]);continue
        # Learned layouts keep intended overlaps (text on shapes/photos); also compete unrepaired.
        # Solved learned layouts already satisfy every check: assess them as they are.
        if template.startswith(('learned_s0','learned_strip')):evaluate(template,name,desc,design,[]);continue
        if template.startswith('learned_'):evaluate(template+'_raw',name,desc,design,[])
        fixed,trace=repair(source,_lock(design,A,locks),A,locks)
        evaluate(template,name,desc,fixed,trace)
    def rank(r):
        # Rubric gates (docs/chameleon-design-guardrails-and-eval-rubric.md) once computed.
        rub=r.get("rubric");gate=rub["gate"] if rub else True
        hard=r["quality"]["defect_count"]+(len(rub["tier2"]["defects"]) if rub and rub["tier2"] else 0)
        overall=rub["tier2"]["overall"] if rub and rub["tier2"] else r["score"]["total"]
        from .refine import blocking
        return (r["valid"] and gate,-(blocking(r) if rub else 0),-hard,1 if r["template"]=="identity" else 0,preference_score(r) if preference_score(r) is not None else overall)
    results.sort(key=rank,reverse=True)
    reference=next((r for r in results if not r['template'].startswith('retrieved')),None)
    if reference:
        ref={e['id']:e for e in reference['design']['elements']}
        def prominent(r):
            for e in r['design']['elements']:
                role=A.roles.get(e['id']);before=ref[e['id']]
                if role in ('headline','cta') and e['type']=='text' and e['font_size']<.7*before['font_size']:return False
                if role in ('hero','logo') and e['width']*e['height']<.49*before['width']*before['height']:return False
            return True
        results=[r for r in results if not r['template'].startswith('retrieved') or prominent(r)]
    if options.get("solver",True):
        for r in results[:3]:
            solved,event=solve(source,r["design"],A,locks,mode=options.get('solver_mode','legacy'))
            if solved:
                solved,trace=repair(source,solved,A,locks)
                evaluate("solver_"+r["template"],"Optimized "+r["name"],r["description"],solved,[{"solver":event}]+trace)
        results.sort(key=rank,reverse=True)
    for r in results:finish(r)
    results.sort(key=rank,reverse=True)
    if options.get("refine",True):
        # Evaluate-and-fix: hill-climb the three best distinct candidates (engine/refine.py).
        from .refine import refine
        seeds=[];fams=set()
        for r in results:
            fam=r["template"].replace("solver_","").split("_scale")[0]
            if r["valid"] and r["rubric"]["gate"] and fam not in fams:seeds.append(r);fams.add(fam)
            if len(seeds)==3:break
        end=time.monotonic()+float(options.get("refine_seconds",12))
        for n,seed in enumerate(seeds):
            def assess(design,seed=seed):
                r=build("refined_"+seed["template"],"Refined "+seed["name"],seed["description"],design,[])
                return finish(r) if r else None
            share=time.monotonic()+(end-time.monotonic())/(len(seeds)-n)
            best,steps=refine(seed,assess,source,A,locks,int(options.get("refine_rounds",10)),share,float(options.get("refine_target",95)),
                              mode=options.get("refinement_mode"),photo_mode=options.get("photo_mode"),
                              allow_solver=options.get('solver',True))
            if steps:best["trace"]=seed["trace"]+steps;results.append(best)
        results.sort(key=rank,reverse=True)
    if options.get("render",True):
        # Verify newly promoted winners too: pixel evidence can change ranking.
        verified_ids=set()
        while True:
            results=_prefer_learned(results,rank,source,A)
            pending=[r for r in results[:max(1,min(3,top_k))] if r['id'] not in verified_ids]
            if not pending:break
            for r in pending:
                verified_ids.add(r['id'])
                r["diagnostics"]=diagnose(source,r["design"],A,pixels=True);finish(r)
            results.sort(key=rank,reverse=True)
    for r in results:
        r["trace"].append({"roles":role_status,"events":events,"elapsed_ms":round((time.monotonic()-started)*1000)})
    results.sort(key=rank,reverse=True)
    results=_prefer_learned(results,rank,source,A)
    if all_candidates:return results
    lead=results[:1] if results and results[0].get('preferred_learned') else []
    if _provider_workflow(options):
        from .providers import fingerprint, mmr_select
        if lead:
            rest=mmr_select(results[1:],top_k-1,quality=lambda r:r['rubric']['tier2']['overall'] if r['rubric'].get('tier2') else r['score']['total'],
                            fingerprint_of=lambda r:fingerprint(r['design'],A),gate=lambda r:rank(r)[:3]) if top_k>1 else []
            return lead+rest
        return mmr_select(results,top_k,quality=lambda r:r['rubric']['tier2']['overall'] if r['rubric'].get('tier2') else r['score']['total'],
                          fingerprint_of=lambda r:fingerprint(r['design'],A),gate=lambda r:rank(r)[:3])
    picked=[];families=set()
    for r in results:
        family=r["template"].replace("refined_","").replace("solver_","").split("_scale")[0]
        if family not in families and (not picked or rank(r)[:3]==rank(picked[0])[:3] and r["score"]["total"]>=picked[0]["score"]["total"]-12):
            picked.append(r);families.add(family)
        if len(picked)>=top_k:break
    for r in results:
        if len(picked)>=top_k:break
        if r not in picked:picked.append(r)
    return lead+sorted([r for r in picked if r not in lead],key=rank,reverse=True)


LEARNED_SOFT_DEFECTS={'scene_displacement'}


def _content_share(design):
    """Share of the canvas spanned by the content (backgrounds that cover the canvas excluded)."""
    from .canonical import rotated_extent, union_all
    cw,ch=float(design['canvas']['width']),float(design['canvas']['height'])
    boxes=[rotated_extent(e).clip(cw,ch) for e in design['elements']]
    boxes=[b for b in boxes if b.w*b.h<.97*cw*ch and b.w>0 and b.h>0]
    return union_all(boxes).area/(cw*ch) if boxes else 1.0


def _new_occlusion(source,design,limit=.25):
    """Ids of elements newly hidden: covered by later-drawn non-text elements over `limit` more of
    their area than in the source. Covers the decorations the block collision check skips (a
    panel slid over a portrait). Canvas-covering backgrounds are not occluders."""
    from .canonical import rotated_extent
    def covered(d):
        cw,ch=float(d['canvas']['width']),float(d['canvas']['height'])
        els=sorted(d['elements'],key=lambda e:e.get('z_index',0))
        box={e['id']:rotated_extent(e) for e in els}
        out={}
        for k,e in enumerate(els):
            b=box[e['id']]
            if b.area<=0 or b.area>=.97*cw*ch:continue
            # Largest single occluder: union areas of overlapping occluders would double count.
            out[e['id']]=max([box[o['id']].intersection_area(b)/b.area for o in els[k+1:]
                              if o['type']!='text' and box[o['id']].area<.97*cw*ch],default=0.0)
        return out
    before,after=covered(source),covered(design)
    return [i for i,f in after.items() if f>before.get(i,0.0)+limit]


def _collapse_floor(design):
    """Share of the source's content share a learned layout must keep. Extreme strips (4:1 and
    beyond) are a single row or column with margins, so they keep less by construction."""
    a=design['canvas']['width']/design['canvas']['height']
    return .3 if a>=4 or a<=.25 else .5


def _learned_eligible(r,source=None):
    """A learned layout is used unless a hard check vetoes it, its content has collapsed into a
    small part of the canvas (under half the share the source used), or it newly hides an
    element under another."""
    from .refine import blocking
    if not (r['valid'] and r.get('rubric',{}).get('gate') and blocking(r)==0):return False
    if source is not None and (_content_share(r['design'])<_collapse_floor(r['design'])*_content_share(source) or _new_occlusion(source,r['design'])):
        return False
    defects={k for k,v in r['quality']['defects'].items() if v}
    return not (defects-LEARNED_SOFT_DEFECTS) and not r['diagnostics'].get('collisions') and not r['diagnostics'].get('contrast_regressions')


def _appeal_selection():
    import os
    from .retarget import appeal
    return os.environ.get('CHAMELEON_POSTPROC','')=='solver' and appeal.available()


def _prefer_learned(results,rank,source=None,A=None):
    """Designer-trained layouts win over templates whenever they pass every hard check; among
    them the rubric decides. Otherwise the existing ranking stands."""
    for r in results:r.pop('preferred_learned',None)
    if source is not None:
        # Learned layouts that collapse or hide an element rank below every other candidate;
        # the rubric does not see those faults.
        bad=lambda r:'learned_' in r['template'] and (_content_share(r['design'])<_collapse_floor(r['design'])*_content_share(source) or bool(_new_occlusion(source,r['design'])))
        results=[r for r in results if not bad(r)]+[r for r in results if bad(r)]
    learned=[r for r in results if 'learned_' in r['template'] and _learned_eligible(r,source)]
    if not learned:return results
    if source is not None and _appeal_selection():
        # Constrained decoding is on: every eligible learned layout is valid, so people's
        # preference (the learned appeal score) picks among them.
        from .retarget import appeal
        A=A or analyze(source)
        si=appeal.source_info(source,A)
        for r in learned:
            if 'appeal' not in r:
                t2=r.get('rubric',{}).get('tier2') or {}
                try:r['appeal']=appeal.score(source,r['design'],A,si,judged=(True,[],{'overall':t2.get('overall'),'scores':t2.get('scores')}))
                except Exception:r['appeal']=0.0
        best=max(learned,key=lambda r:(r['appeal'],not r['template'].startswith(('refined_','solver_'))));best['preferred_learned']=True
        return [best]+[r for r in results if r is not best]
    # Trust the model's own most likely layout first (greedy sample 0, unrepaired), then its
    # repaired form, then its copy-bound fallback, then other samples; engine refinements of a
    # learned layout come last.
    def order(r):
        t=r['template']
        # learned_s0: the constrained decoding of the greedy layout (CHAMELEON_POSTPROC=solver)
        return (t.startswith(('refined_','solver_')),not t.startswith(('learned_s0','learned_0')),not t.startswith('learned_s0'),'_bound' in t,not t.endswith('_raw'),t)
    best=min(learned,key=order);best['preferred_learned']=True
    return [best]+[r for r in results if r is not best]


def assess_design(source, target, design, A, name='Edited design', locks=None):
    """Shared deterministic assessment of a materialized controller revision."""
    from .quality import diagnose
    from .judging import judge_quality
    from .rubric import evaluate
    import hashlib
    design={'canvas':dict(target),'elements':copy.deepcopy(design['elements'])}
    checks=validate(source,design)
    by={e['id']:e for e in design['elements']}
    for eid,rect in (locks or {}).items():
        ok=eid in by and all(abs(by[eid].get(k,-1e9)-v)<=.5 for k,v in rect.items())
        checks.append({'name':'Locked placement: '+eid,'passed':ok,'severity':'error','detail':'Exact placement' if ok else 'Lock mismatch'})
    placement=_placement(A,design,name)
    q=score(source,design,A,placement)
    diagnostics=diagnose(source,design,A)
    quality=judge_quality(source,design,A,diagnostics)
    rubric=evaluate(source,design,A,diagnostics,q,target)
    return {'id':hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()[:16],
            'template':'controller','name':name,'description':'Iteratively adapted with verified checkpoints',
            'design':design,'checks':checks,'valid':passes(checks),'score':q,'quality':quality,
            'rubric':rubric,'diagnostics':diagnostics,'overlay':placement.overlay,'trace':[]}


def _controller_seeds(source,target,A,emphasis,options,locks):
    """Cheap structural initialization; alternatives are consumed only on restart."""
    from .repair import repair
    from .scene import propose_photo_scene
    if source['canvas']==target:
        yield 'Original composition',_lock(copy.deepcopy(source),A,locks)
    for design in propose_photo_scene(source,target,A):
        yield 'Photo with copy column',_lock(design,A,locks)
    placements=propose(A,target['width'],target['height'],emphasis)
    ratio=target['width']/target['height']
    def priority(p):
        base=p.template.split('_scale')[0]
        preferred=('banner' in base if ratio>=5 else 'split' in base if ratio>=1.8 else 'stack' in base)
        return (not preferred,'_scale' in p.template,'columns' in base)
    for placement in sorted(placements,key=priority):
        design=serialise(source,target,placement)
        design,_=repair(source,_lock(design,A,locks),A,locks)
        yield placement.name,design
    if options.get('retrieval',True):
        from .retrieval import retrieve
        try:
            for design in retrieve(source,target,A):yield 'Retrieved arrangement',_lock(design,A,locks)
        except (ValueError,OSError,KeyError):pass
    # Rotation-aware proportional source-frame remains a recovery option.
    from .canonical import union_all,rotated_extent,Box
    frame=union_all([rotated_extent(e) for e in source['elements']]+[Box(0,0,A.sw,A.sh)])
    scale=min(target['width']/frame.w,target['height']/frame.h)
    ox=(target['width']-frame.w*scale)/2-frame.x*scale
    oy=(target['height']-frame.h*scale)/2-frame.y*scale
    design=copy.deepcopy(source);design={'canvas':dict(target),'elements':design['elements']}
    for e in design['elements']:
        e.update(x=round(e['x']*scale+ox,2),y=round(e['y']*scale+oy,2),width=round(e['width']*scale,2),height=round(e['height']*scale,2))
        if e['type']=='text':e['font_size']=max(1,int(e['font_size']*scale))
    yield 'Original arrangement',_lock(design,A,locks)


def generate(source,target,emphasis='balanced',top_k=3,analysis=None,*,locks=None,
             options=None,all_candidates=False,progress_callback=None,artifact_callback=None,
             cancelled=None):
    """Compatibility boundary for legacy batches and bounded one-design sessions."""
    from .contracts import Budget,CONTROLLER_MODES,design_hash
    options=dict(options or {});locks=locks or {}
    mode=options.get('controller_mode','legacy_batch')
    if mode not in CONTROLLER_MODES:raise ValueError('Unknown controller mode: '+str(mode))
    if mode=='legacy_batch':
        return _generate_legacy(source,target,emphasis,top_k,analysis,locks=locks,options=options,all_candidates=all_candidates)
    from .controller import run_controller
    from .quality import verify_design,diagnose
    from .refine import OPERATORS
    from .learning import apply_roles
    from .judging import judge_quality
    from .rubric import evaluate
    from .constraints import solve
    A=copy.deepcopy(analysis) if analysis else analyze(source,group_policies=options.get("group_policies"))
    role_status=apply_roles(source,A,enabled=options.get('role_mode',options.get('roles')))
    budget=Budget(seconds=options.get('controller_seconds',12.0),max_edits=options.get('max_edits',24),
                  max_renders=options.get('max_renders',8),max_model_calls=options.get('max_model_calls',4),
                  cancellation=cancelled)
    assessments={};rendered_images={};source_image=[];learning_events=[]
    def assess(design,name):
        r=assess_design(source,target,design,A,name,locks)
        assessments[design_hash(design)]=r
        return r
    def verify(design):
        verification=verify_design(design,timeout=max(.05,min(12,budget.remaining())))
        if verification.get('verified') and options.get('visual_judge','off')=='openjev' and budget.model_calls<budget.max_model_calls:
            evidence=verification['image'].copy();evidence.thumbnail((1024,1024))
            rendered_images[design_hash(design)]=evidence
        if verification.get('verified') and budget.remaining()>.1:
            r=assessments[design_hash(design)]
            # Full output is verified independently of approximate contrast.
            # Prefix diagnostics remain bounded by their own worker deadline.
            if options.get('pixel_diagnostics',False):
                r['diagnostics']=diagnose(source,design,A,pixels=True,budget=budget)
                r['quality']=judge_quality(source,design,A,r['diagnostics'])
                r['rubric']=evaluate(source,design,A,r['diagnostics'],r['score'],target)
        return verification
    def expand(candidate,context_budget):
        if options.get('refine',True) and options.get('refinement_mode')=='beam_lns':
            from .edit_operators import expand as coordinated_edits
            yield from coordinated_edits(candidate,context_budget,source=source,analysis=A,locks=locks,
                                         photo_mode=options.get('photo_mode'),allow_solver=options.get('solver',True))
        elif options.get('refine',True):
            for operator in OPERATORS:
                if context_budget.stop_reason():return
                for name,design in operator(source,candidate,A,set(locks)):
                    yield name,design
        if options.get('solver',True) and context_budget.remaining()>.05:
            design,event=solve(source,candidate['design'],A,locks,mode=options.get('solver_mode','legacy'),
                               budget_seconds=min(.4,context_budget.remaining()),cancelled=cancelled)
            if design:yield 'Constraint optimization ('+event.get('status','')+')',design
        if (options.get('learned_edit',False) and not context_budget.stop_reason()
                and context_budget.model_calls<context_budget.max_model_calls
                and context_budget.renders+3<=context_budget.max_renders):
            from .contracts import DesignRevision
            from .materialize import apply_edit
            from .neural_proposals import propose_edit
            revision=DesignRevision.create(candidate['design'])
            context_budget.model_calls+=1;context_budget.renders+=2
            try:
                proposal,event=propose_edit(source,target,A,revision,candidate['rubric'].get('tier2'),locks,
                    timeout=min(30,context_budget.remaining()),cancelled=cancelled)
            except Exception as exc:
                proposal=None;event={'status':'model_failed','detail':type(exc).__name__}
            context_budget.model_calls+=int(event.get('model_calls',1))-1
            context_budget.renders+=int(event.get('renders',2))-2
            learning_events.append(event)
            if progress_callback:progress_callback({'type':'learned_edit','status':event.get('status'),
                                                    'revision_hash':revision.revision_hash})
            if proposal is not None:
                try:design=apply_edit(source,revision,proposal,locks)
                except (ValueError,KeyError,TypeError) as exc:
                    learning_events.append({'status':'materialization_rejected','detail':str(exc)})
                else:yield 'Learned edit: '+proposal.operation,design
    compare=None
    if options.get('visual_judge','off')=='openjev':
        from .evaluators import OpenJevEvaluator
        # Audit position swaps separately; ordinary edit comparisons cost one
        # model decision. Source/before/after images are cached and all newly
        # rendered source evidence is charged to the session's render budget.
        evaluator=OpenJevEvaluator(swap=False)
        def compare(before,after):
            if not source_image:
                if budget.renders>=budget.max_renders:
                    return {'available':False,'reason':'source_render_budget','provenance':{'model_calls':0}}
                budget.renders+=1
                original=verify_design(source,timeout=max(.05,min(12,budget.remaining())))
                if not original.get('verified'):
                    return {'available':False,'reason':original.get('error'),'provenance':{'model_calls':0}}
                evidence=original['image'].copy();evidence.thumbnail((1024,1024));source_image.append(evidence)
            images=[source_image[0],rendered_images[design_hash(before)],rendered_images[design_hash(after)]]
            verdict=evaluator.compare(source,target,before,after,images=images,timeout=max(.05,min(15,budget.remaining())))
            keep={design_hash(before),design_hash(after)}
            for key in list(rendered_images):
                if key not in keep:rendered_images.pop(key)
            return verdict
    initializer=None
    seeds=_controller_seeds(source,target,A,emphasis,options,locks)
    if _provider_workflow(options) or options.get('vlm'):
        from .providers import Initializer,ProposalContext
        initializer=Initializer(portfolio=options.get('proposal_portfolio'),retrieval_version=options.get('retrieval_version'))
        context=ProposalContext(copy.deepcopy(source),dict(target),A,locks,dict(options),emphasis,role_status=role_status)
        seeds=initializer(context,budget)
    try:
        result=run_controller(source,seeds,assess=assess,
                              expand=expand,verify=verify,mode=mode,locks=locks,budget=budget,
                              target_score=float(options.get('controller_target',95)),compare=compare,
                              progress_callback=progress_callback,artifact_callback=artifact_callback)
    except Exception as exc:
        from .contracts import NoValidDesign
        if isinstance(exc,NoValidDesign):
            exc.report.update(providers=[r.to_dict() for r in initializer.results] if initializer else [],
                              learned_edits=learning_events)
        raise
    result[0]['trace'].append({'roles':role_status,'providers':[r.to_dict() for r in initializer.results] if initializer else [],
                              'learned_edits':learning_events})
    return result


def analyze_source(source):
    return analyze(source).to_dict()


def best(source,target):
    candidates=generate(source,target,top_k=1)
    if candidates:
        top=candidates[0]
        if not (top["valid"] and top["rubric"]["gate"]):
            from .contracts import NoValidDesign
            raise NoValidDesign('Best candidate fails structural guardrails',report=top.get('rubric'))
        return top["design"]
    raise ValueError("No candidate layout could be generated")
