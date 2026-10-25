"""Versioned bounded rewards with independent raw/repaired validity accounting.

This deterministic rubric is an auxiliary training signal, never human preference
or independent proof of design improvement. Runtime verification remains mandatory.
"""
import hashlib,json,math,time
from engine.analysis import analyze
from engine.metrics import WEIGHTS
from engine.neural_proposals import materialize_plan,parse_plan_text
from engine.pipeline import _placement
from engine.rubric import evaluate
REWARD_VERSION='bounded-preservation-v3'


def assess(source,design,A=None,target=None):
    from engine.metrics import score
    from engine.quality import diagnose
    from engine.invariants import validate,passes
    A=A or analyze(source);checks=validate(source,design)
    diagnostics=diagnose(source,design,A);quality=score(source,design,A,_placement(A,design))
    return {'design':design,'valid':passes(checks),'checks':checks,'score':quality,'diagnostics':diagnostics,
            'rubric':evaluate(source,design,A,diagnostics,quality,target or design['canvas'])}


def reward_from(result,preference=None,lam=.2):
    rub=result['rubric']
    if not rub['gate'] or not rub['tier2']:return -1.
    if isinstance(lam,bool) or not isinstance(lam,(int,float)) or not math.isfinite(lam) or lam<0:raise ValueError('Preference weight must be finite and nonnegative')
    t2=rub['tier2'];parts=result['score']['parts'];stats=t2['stats']
    gate=.25 if t2['defects'] else 1.
    base=gate*sum(WEIGHTS[k]*parts[k] for k in WEIGHTS)/100
    margins=sum(w['id']=='space.margin' for w in t2['warnings'])
    value=base+.15*stats.get('aligned_share',1)-.10*min(1,stats.get('near_misses',0)/4)-.10*min(1,margins/3)
    if preference is not None and not isinstance(preference,bool) and isinstance(preference,(int,float)) and math.isfinite(preference):
        value+=min(.2,lam)*max(-1.,min(1.,preference))
    return round(max(0.,min(1.,value)),4)


def parse_plan(text):
    if isinstance(text,list):
        text=text[-1].get('content','') if text and isinstance(text[-1],dict) else ''
        if isinstance(text,list):text=''.join(item.get('text','') for item in text if isinstance(item,dict))
    return parse_plan_text(text) if isinstance(text,str) else text if isinstance(text,dict) else None


def version_identity():
    from pathlib import Path
    from engine.canonical import ROOT
    paths=['engine/neural_proposals.py','engine/invariants.py','engine/rubric.py','engine/metrics.py','engine/quality.py','engine/analysis.py','engine/relations.py','training/reward.py']
    files={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
    return {'reward_version':REWARD_VERSION,'materializer_schema':['layout-plan/1','layout-plan/2'],
            'source_hashes':files,'identity_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'analysis_graph':getattr(__import__('engine.relations',fromlist=['GRAPH_VERSION']),'GRAPH_VERSION','unversioned'),'preference_origin':'deterministic_surrogate; no human model assumed'}


def reward_record(source,target,plan,A=None,*,locks=None,preference=None,repair=False,repair_seconds=.1,cancelled=None):
    """A bounded whole-plan episode. Repaired validity never conceals a bad raw plan."""
    from engine.contracts import design_hash
    from engine.invariants import validate,passes
    started=time.monotonic();A=A or analyze(source)
    result={'reward_version':REWARD_VERSION,'reward':-1.,'bounds':[-1.,1.],'structural_failure':True,'raw_valid':False,
            'repaired_valid':False,'channels':{},'repairs':{'enabled':bool(repair),'shift_px':None,'events':[]},
            'source_hash':design_hash(source),'target':dict(target),'plan_sha256':None,'model_calls':0,'renders':0}
    def done(status):
        result.update(status=status,seconds=round(time.monotonic()-started,6));return result
    if cancelled and cancelled():return done('cancelled')
    try:result['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest()
    except (ValueError,TypeError):return done('invalid_plan')
    outcome=materialize_plan(source,target,A,plan,locks)
    result['materialization']=outcome.to_dict();result['raw_valid']=outcome.raw_valid;result['repaired_valid']=outcome.repaired_valid
    if outcome.raw_design is not None:
        raw=assess(source,outcome.raw_design,A,target);result['raw_assessment']={k:v for k,v in raw.items() if k!='design'}
        result['raw_revision']=design_hash(outcome.raw_design)
    if outcome.design is None:return done(outcome.status)
    design=outcome.design
    if repair and not (cancelled and cancelled()):
        # Existing repair is deterministic and bounded; record its burden separately.
        if isinstance(repair_seconds,bool) or not math.isfinite(repair_seconds) or repair_seconds<0:raise ValueError('invalid repair budget')
        from engine.repair import repair as repair_layout
        before={e['id']:e for e in design['elements']}
        if repair_seconds>0:
            repair_started=time.monotonic();design,events=repair_layout(source,design,A,locks or {})
            result['repairs']['seconds']=round(time.monotonic()-repair_started,6)
            result['repairs']['deadline_scope']='cooperative fixed-pass repair; no in-call preemption'
            if time.monotonic()-repair_started>repair_seconds:
                result['repairs']['events']=events;return done('repair_budget')
            result['repairs']['events']=events
        result['repairs']['shift_px']=max((abs(before[e['id']][key]-e[key]) for e in design['elements'] for key in ('x','y','width','height')),default=0.)
        result['repairs']['budget_seconds']=repair_seconds
    if cancelled and cancelled():return done('cancelled')
    repaired=assess(source,design,A,target);result['repaired_valid']=bool(repaired['valid'] and repaired['rubric']['gate'])
    result['repaired_revision']=design_hash(design);result['repaired_assessment']={k:v for k,v in repaired.items() if k!='design'}
    result['structural_failure']=not (outcome.raw_valid and result['repaired_valid'])
    if result['structural_failure']:return done('structural_failure')
    base=reward_from(repaired);burden=min(.1,float(result['repairs'].get('shift_px') or 0)/max(1,max(target.values()))*.1)
    shaping=0.
    if preference is not None and not isinstance(preference,bool) and isinstance(preference,(int,float)) and math.isfinite(preference):shaping=.1*max(-1.,min(1.,preference))
    result['channels']={'deterministic_quality':base,'bounded_auxiliary_preference':shaping,'repair_burden_penalty':burden}
    result['reward']=round(max(0.,min(1.,base+shaping-burden)),4)
    return done('accepted')


class RewardCache:
    """Bounded deterministic episode cache; no model or human predictions cached implicitly."""
    def __init__(self,max_entries=128):
        from collections import OrderedDict
        self.entries=OrderedDict();self.max_entries=max_entries
    def evaluate(self,source,target,plan,A=None,*,analysis_options=None,**kwargs):
        import copy
        from engine.quality import resource_fingerprint
        cancelled=kwargs.get('cancelled')
        if kwargs.get('repair') or (cancelled and cancelled()):return reward_record(source,target,plan,A,**kwargs)
        context={'source':source,'target':target,'plan':plan,'analysis_options':analysis_options or {},
                 'locks':kwargs.get('locks'),'preference':kwargs.get('preference'),'versions':version_identity(),
                 'resources':resource_fingerprint(source)}
        try:key=hashlib.sha256(json.dumps(context,sort_keys=True,allow_nan=False).encode()).hexdigest()
        except (ValueError,TypeError):return reward_record(source,target,plan,A,**kwargs)
        if key in self.entries:
            self.entries.move_to_end(key);record=copy.deepcopy(self.entries[key]);record.update(cache_hit=True,seconds=0.,model_calls=0,renders=0);return record
        record=reward_record(source,target,plan,A,**kwargs)
        if record['status'] not in ('cancelled','repair_budget'):
            self.entries[key]=copy.deepcopy(record)
            while len(self.entries)>self.max_entries:self.entries.popitem(last=False)
        return record


def reward(source,target,plan,A=None,preference=None,locks=None):
    return reward_record(source,target,plan,A,preference=preference,locks=locks)['reward']


def grpo_reward(prompts,completions,task,**kwargs):
    if len(completions)!=len(task):raise ValueError('completion/task count mismatch')
    out=[]
    for completion,record in zip(completions,task):
        record=json.loads(record) if isinstance(record,str) else record
        out.append(reward(record['source'],record['target'],parse_plan(completion),locks=record.get('locks')))
    return out
