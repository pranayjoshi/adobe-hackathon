"""Isolated bounded visual policy worker. Resources are verified, never substituted."""
import json,sys,time
from pathlib import Path
from .vlm import describe,generate_text,status
from . import constitution
from .quality import verify_design


def run(request,answer):
    req=json.loads(Path(request).read_text());answer=Path(answer);started=time.monotonic()
    event={'status':'initializing','model_calls':0,'renders':0};deadline=started+float(req.get('timeout_seconds',90))
    def save(**fields):
        event.update(fields);tmp=answer.with_suffix('.tmp');tmp.write_text(json.dumps(event,allow_nan=False));tmp.replace(answer)
    try:
        task=req.get('task','plan');schema_id=req.get('schema','layout-plan/1')
        schemas={'layout-plan/1':constitution.PLAN_SCHEMA,
                 'layout-plan/2':getattr(constitution,'PLAN_SCHEMA_V2',None),
                 'edit-action/1':getattr(constitution,'EDIT_ACTION_SCHEMA',None)}
        schema=schemas.get(schema_id)
        if task not in ('plan','edit') or schema is None:save(status='unsupported_schema');return
        images=[]
        designs=[req['source']]+([req['state']] if task=='edit' else [])
        for design in designs:
            remaining=deadline-time.monotonic()
            if remaining<=0:save(status='timeout');return
            save(status='rendering',renders=event['renders']+1)
            verification=verify_design(design,timeout=max(.01,min(12,remaining)),max_size=672)
            if verification.get('cache_hit'):event['renders']-=1
            if not verification.get('verified'):
                save(status='resource_or_render_unavailable',detail=verification.get('error'));return
            images.append(verification['image'])
        if time.monotonic()>=deadline:save(status='timeout');return
        save(status='generating',model_calls=1)
        if status().get('backend')=='transformers':
            raw=generate_text(images,req['prompt'],max_new_tokens=int(req.get('max_new_tokens',768)))
            save(status='generated' if raw is not None else 'model_failed',raw=raw)
        else:
            plan=describe(images,req['prompt'],schema)
            save(status='generated' if plan else 'model_failed',plan=plan)
    except Exception as exc:save(status='model_failed',detail=type(exc).__name__+': '+str(exc))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
