"""Direct preference optimization for whole-plan and edit-action policies (guide 16, plan 20 C2).

Pipeline (each step is a CLI subcommand; nothing downloads, nothing is a human vote unless reviewed):

  # preference records (chameleon-pref/1) from three possible origins
  python -m training.edit_dpo synthesize output/learn/demos.jsonl --out output/learn-dpo/prefs.synthetic.jsonl
  python -m training.edit_dpo import-legacy $REPO/output/training/vlm-smoke --out output/learn-dpo/prefs.legacy.jsonl
  #   (reviewed human comparisons arrive as chameleon-pref/1 records with provenance explicit_human + review{})
  # records -> binary DPO datasets (one directory per provenance variant) + non-binary store + exclusions
  python -m training.edit_dpo build output/learn-dpo/prefs.*.jsonl --out output/learn-dpo/data --split-manifest output/learn/data/split_manifest.json
  python -m training.edit_dpo validate output/learn-dpo/data/synthetic_only
  # train from an exported SFT bundle; the frozen reference is a copy of that SFT adapter
  python -m training.edit_dpo train output/learn-dpo/data/synthetic_only --sft-bundle B --out OUT --beta 0.1 --lr 1e-5 --max-steps 4 --device cpu
  python -m training.edit_dpo sweep-spec | experiment-spec

Preference record (input, ``chameleon-pref/1``)::

  {pref_schema, pref_id, kind: plan|edit, provenance, teacher|review{reviewer_id,committed_at,verdict}|event_id,
   source_family, design_id, verdict: A|B|tie|both_bad|unsure,
   context: {source, target, locks?, state? (edit: the base design), state_defects?, plan_schema?},
   presentation: {A: candidate_id, B: candidate_id},          # what the reviewer saw on each side
   candidates: {candidate_id: {completion: str} | {design: {...}},  # plus optional per-side context keys
               ...}}                                            # (source_hash, target, locks, base_revision, image)

Binary rows exist only for decisive verdicts (A/B). tie/both_bad/unsure are written to ``nonbinary.jsonl`` and
counted, never turned into chosen/rejected rows. Orientation is resolved from ``presentation[verdict]`` so A/B
display randomization cannot flip a pair. Both sides are re-parsed, canonicalized and materialized through the
runtime path (``materialize_plan`` / ``action_to_proposal`` + ``engine.materialize.apply_edit``).

Surrogate-reward audit: synthetic/teacher pairs must agree with ``training.reward`` (else excluded, as before);
a decisive human (or behavioral) pair whose chosen side has the lower surrogate reward is KEPT and flagged
``reward_disagreement`` (counted in the audit). The surrogate is not ground truth for human preference.
"""
import argparse,hashlib,json,math,random,statistics,time
from collections import Counter
from pathlib import Path
from engine.analysis import analyze
from engine.contracts import DesignRevision,design_hash
from engine.neural_proposals import (PLAN_V1,PLAN_V2,EDIT_V1,PlanRejected,action_to_proposal,design_to_plan,design_to_plan_v2,
                                     edit_action,edit_prompt,materialize_plan,parse_plan_text,plan_fidelity,plan_prompt,plan_prompt_v2)
from training.edit_dataset import (EXPORT_SPLITS,IMAGE_TRANSFORM,PROVENANCE,_dumps,_render,build_split_manifest,check_provenance,
                                   excluded,resolve_family,sha256_text,split_of,suite_exclusions)

PREF_SCHEMA='chameleon-pref/1'
DPO_DATASET_SCHEMA='chameleon-dpo-dataset/1'
VERDICTS=('A','B','tie','both_bad','unsure')
DECISIVE=('A','B')
HUMAN=('explicit_human','behavioral')           # surrogate disagreement is audited, never filtered
SYNTHETIC=('teacher_model','deterministic_synthetic')  # surrogate ordering is still enforced
VARIANTS={'human_only':('explicit_human',),
          'human_plus_synthetic':('explicit_human','teacher_model','deterministic_synthetic'),
          'synthetic_only':('teacher_model','deterministic_synthetic')}
CANON_OPERATION='apply_patch'   # edit completions: the operation label carries no preference signal
NEAR_PX=1.0                     # sides closer than this (and same font sizes) are not a real preference
DUP_GRID_PX=2.0                 # near-duplicate pair signature quantization
SIDE_CONTEXT_KEYS=('source_hash','target','locks','base_revision','image','plan_schema')
FILES={('plan','train'):'plan_dpo.jsonl',('plan','validation'):'plan_dpo.validation.jsonl',
       ('edit','train'):'edit_dpo.jsonl',('edit','validation'):'edit_dpo.validation.jsonl'}


# ------------------------------------------------------------- canonical sides

def canonical_completion(kind,obj):
    """Canonical JSON for one side: sorted keys, compact separators, original IDs, no free text.

    Plans drop ``rationale``; edit actions drop ``intended_improvement``/``affected_ids`` and use one fixed
    operation label, so neither verbosity nor a teacher's operation name can become the learned distinction."""
    if kind=='plan':
        obj={k:v for k,v in obj.items() if k!='rationale'}
    else:
        obj={'schema':EDIT_V1,'operation':CANON_OPERATION,'patch':obj.get('patch')}
    return _dumps(obj)


def materialize_side(kind,context,completion,A=None):
    """Parse -> typed validation -> exact materialization (runtime path). Returns (status, design, reasons)."""
    source,target,locks=context['source'],context['target'],context.get('locks') or None
    obj=parse_plan_text(completion)
    if obj is None:return 'unparseable',None,['unparseable']
    if kind=='plan':
        o=materialize_plan(source,target,A or analyze(source,group_policies=(context.get('options') or {}).get('group_policies')),obj,locks)
        return (o.status if not (o.status=='ok' and not o.raw_valid) else 'raw_invalid',o.design,[r['code'] for r in o.reasons][:8])
    rev=DesignRevision.create(context['state'])
    try:
        from engine.materialize import apply_edit
        return 'ok',apply_edit(source,rev,action_to_proposal(obj,rev,provenance='model',locks=locks),locks),[]
    except PlanRejected as exc:return 'rejected',None,[r['code'] for r in exc.reasons][:8]
    except (ValueError,TypeError) as exc:return 'materialization_rejected',None,[str(exc)[:80]]


def side_from_design(kind,context,design,A=None):
    """A reviewed design (e.g. a rendered candidate) -> canonical completion that reproduces it, or None."""
    if design['canvas']!={'width':int(context['target']['width']),'height':int(context['target']['height'])}:return None
    if kind=='plan':
        schema=context.get('plan_schema',PLAN_V2)
        A=A or analyze(context['source'],group_policies=(context.get('options') or {}).get('group_policies'))
        plan=design_to_plan_v2(design,context['source'],A) if schema==PLAN_V2 else design_to_plan(design,A)
        return canonical_completion('plan',plan)
    action=edit_action(context['state'],design,CANON_OPERATION)
    return canonical_completion('edit',action) if action['patch'] else None


def context_hash(kind,context):
    keys={'kind':kind,'source_hash':design_hash(context['source']),'target':context['target'],'locks':context.get('locks') or {},
          'image_transform':IMAGE_TRANSFORM,'schema':context.get('plan_schema',PLAN_V2) if kind=='plan' else EDIT_V1}
    if (context.get('options') or {}).get('group_policies') is not None:keys['group_policies']=context['options']['group_policies']
    if kind=='edit':keys['base_revision']=design_hash(context['state'])
    return sha256_text(_dumps(keys))


def side_context_mismatch(kind,context,cand):
    """Names of per-side context keys that disagree with the record's shared context."""
    want={'source_hash':design_hash(context['source']),'target':context['target'],'locks':context.get('locks') or {},
          'plan_schema':context.get('plan_schema',PLAN_V2)}
    if kind=='edit':want['base_revision']=design_hash(context['state'])
    bad=[k for k in SIDE_CONTEXT_KEYS if k in cand and k in want and cand[k]!=want[k]]
    if 'image' in cand and context.get('image') is not None and cand['image']!=context['image']:bad.append('image')
    return bad


def geometry_gap(a,b):
    """Max geometry difference (px) between two designs and whether font sizes are identical."""
    ga={e['id']:e for e in a['elements']};gb={e['id']:e for e in b['elements']}
    if set(ga)!=set(gb):return math.inf,False
    px=max((abs(float(ga[i][k])-float(gb[i][k])) for i in ga for k in ('x','y','width','height')),default=0.0)
    fonts=all(ga[i].get('font_size')==gb[i].get('font_size') for i in ga)
    return px,fonts and all(ga[i].get('fit')==gb[i].get('fit') for i in ga)


def geometry_signature(design,grid=DUP_GRID_PX):
    rows=sorted((e['id'],*(int(round(float(e[k])/grid)) for k in ('x','y','width','height')),e.get('font_size')) for e in design['elements'])
    return sha256_text(json.dumps(rows))


# -------------------------------------------------------------- reward audit

def surrogate_reward(source,target,design,A=None):
    """training.reward's rubric reward of a materialized design (read-only use of the Codex reward API)."""
    from training.reward import assess,reward_from
    return reward_from(assess(source,design,A or analyze(source,group_policies=(context.get('options') or {}).get('group_policies')),target))


def audit_pair(provenance,r_chosen,r_rejected):
    """'agrees' | 'reward_disagreement' (human: keep, flag, count) | 'misordered' (synthetic: exclude)."""
    if r_chosen is None or r_rejected is None:return 'not_scored'
    if r_chosen>r_rejected:return 'agrees'
    return 'reward_disagreement' if provenance in HUMAN else 'misordered'


# ---------------------------------------------------------- record -> row

def resolve_orientation(record):
    """(chosen_id, rejected_id) for a decisive verdict, else None. Raises ValueError on a broken record."""
    v=record.get('verdict')
    if v not in VERDICTS:raise ValueError('unknown_verdict')
    pres=record.get('presentation') or {}
    if set(pres)!={'A','B'} or pres['A']==pres['B']:raise ValueError('bad_presentation')
    if any(pres[k] not in (record.get('candidates') or {}) for k in 'AB'):raise ValueError('presentation_unknown_candidate')
    if v not in DECISIVE:return None
    return pres[v],pres['B' if v=='A' else 'A']


def record_problems(record):
    out=[]
    if record.get('pref_schema')!=PREF_SCHEMA:out.append('wrong_pref_schema')
    if record.get('kind') not in ('plan','edit'):out.append('unknown_kind')
    for k in ('pref_id','source_family','context','candidates','presentation','verdict'):
        if k not in record:out.append('missing_'+k)
    if out:return out
    out+=check_provenance(record)
    ctx=record['context']
    for k in ('source','target'):
        if k not in ctx:out.append('missing_context_'+k)
    if record['kind']=='edit' and 'state' not in ctx:out.append('missing_context_state')
    if record.get('provenance')=='explicit_human' and (record.get('review') or {}).get('verdict') not in (None,record['verdict']):
        out.append('review_verdict_mismatch')
    return out


def prompt_for(kind,context,A):
    if kind=='plan':
        return plan_prompt_v2(context['source'],context['target'],A,context.get('locks')) if context.get('plan_schema',PLAN_V2)==PLAN_V2 else plan_prompt(context['source'],context['target'],A)
    return edit_prompt(context['source'],context['target'],A,context['state'],context.get('state_defects') or {},context.get('locks'))


def _stats(xs):
    return {'n':len(xs),'mean':round(statistics.fmean(xs),1),'median':statistics.median(xs),'min':min(xs),'max':max(xs)} if xs else {'n':0}


def length_report(rows):
    """Guard: verbosity must not be the learned signal. Length-only accuracy near 0.5 is the healthy case."""
    c=[len(r['chosen']) for r in rows];j=[len(r['rejected']) for r in rows]
    longer=sum(a>b for a,b in zip(c,j));shorter=sum(a<b for a,b in zip(c,j));n=len(rows)
    acc=round(longer/n,3) if n else None
    return {'chosen_chars':_stats(c),'rejected_chars':_stats(j),'chosen_longer':longer,'chosen_shorter':shorter,'equal':n-longer-shorter,
            'length_only_accuracy':acc,'length_bias_warning':bool(n>=20 and acc is not None and (acc>=.7 or acc<=.3)),
            'mean_abs_char_gap':round(statistics.fmean(abs(a-b) for a,b in zip(c,j)),1) if n else None}


def build(pref_paths,out,split_manifest=None,render=True,audit=True,variants=VARIANTS):
    """Preference records -> per-variant binary DPO datasets. Deterministic for identical inputs."""
    out=Path(out);(out/'images').mkdir(parents=True,exist_ok=True);excl=suite_exclusions()
    records=[json.loads(l) for p in pref_paths for l in Path(p).read_text().splitlines() if l.strip()]
    records.sort(key=lambda r:str(r.get('pref_id')))
    if split_manifest is None:
        split_manifest=build_split_manifest([(r.get('design_id') or r.get('source_family'),{'family':r.get('source_family'),'source':(r.get('context') or {}).get('source',{})}) for r in records if r.get('source_family')])
    mtext=_dumps(split_manifest);mhash=sha256_text(mtext);(out/'split_manifest.json').write_text(mtext+'\n')
    counts=Counter();excluded_rows=[];nonbinary=[];binary=[];seen_exact={};seen_near=set();audit_counts=Counter();disagreements=[]
    def drop(record,why,**extra):
        counts['excluded_'+why]+=1;excluded_rows.append({'pref_id':record.get('pref_id'),'kind':record.get('kind'),'provenance':record.get('provenance'),'reason':why,**extra})
    for rec in records:
        counts['records']+=1;counts['provenance_'+str(rec.get('provenance'))]+=1
        probs=record_problems(rec)
        if probs:drop(rec,probs[0],all=probs);continue
        kind,ctx=rec['kind'],rec['context']
        try:orient=resolve_orientation(rec)
        except ValueError as exc:drop(rec,str(exc));continue
        family=resolve_family(rec,rec.get('design_id'))
        why=excluded(excl,source=ctx['source'],family=family,design_id=rec.get('design_id'))
        if why:drop(rec,'visual_suite_'+why);continue
        try:split=split_of(family,split_manifest)
        except ValueError:drop(rec,'family_not_in_manifest');continue
        if split not in EXPORT_SPLITS:drop(rec,'split_'+split);continue
        A=analyze(ctx['source'],group_policies=(ctx.get('options') or {}).get('group_policies'))
        # Both sides: identical context, canonical completion, runtime materialization.
        sides={};bad=None
        for cid,cand in sorted(rec['candidates'].items()):
            mm=side_context_mismatch(kind,ctx,cand)
            if mm:bad=('context_mismatch',{'candidate':cid,'keys':mm});break
            if 'completion' in cand:
                obj=parse_plan_text(cand['completion'])
                comp=canonical_completion(kind,obj) if obj is not None else cand['completion']
            elif 'design' in cand:
                comp=side_from_design(kind,ctx,cand['design'],A)
                if comp is None:bad=('design_not_representable',{'candidate':cid});break
            else:bad=('candidate_without_completion',{'candidate':cid});break
            status,design,reasons=materialize_side(kind,ctx,comp,A)
            if status!='ok':bad=('invalid_side',{'candidate':cid,'status':status,'reasons':reasons});break
            if 'design' in cand and plan_fidelity(cand['design'],design)['class']=='lossy':bad=('lossy_side',{'candidate':cid});break
            sides[cid]={'completion':comp,'design':design,'revision':design_hash(design)}
        if bad:drop(rec,bad[0],**bad[1]);continue
        if orient is None:
            nonbinary.append({'pref_id':rec['pref_id'],'kind':kind,'verdict':rec['verdict'],'provenance':rec['provenance'],'source_family':family,'split':split,
                              'context_hash':context_hash(kind,ctx),'presentation':rec['presentation'],
                              'revisions':{cid:s['revision'] for cid,s in sides.items()},'reasons':rec.get('reasons',[])})
            counts['nonbinary_'+rec['verdict']]+=1;continue
        ch,rj=sides[orient[0]],sides[orient[1]]
        px,same_font=geometry_gap(ch['design'],rj['design'])
        if px<NEAR_PX and same_font:drop(rec,'near_identical_sides',max_px=round(px,3));continue
        chash=context_hash(kind,ctx);key=(chash,ch['revision'],rj['revision'])
        if key in seen_exact:drop(rec,'duplicate_pair',of=seen_exact[key]);continue
        near=(chash,geometry_signature(ch['design']),geometry_signature(rj['design']))
        if near in seen_near:drop(rec,'near_duplicate_pair');continue
        if (chash,rj['revision'],ch['revision']) in seen_exact:counts['conflicting_pairs']+=1  # kept: real disagreement is data
        seen_exact[key]=rec['pref_id'];seen_near.add(near)
        sur={'chosen':None,'rejected':None}
        if audit:
            sur={'chosen':surrogate_reward(ctx['source'],ctx['target'],ch['design'],A),'rejected':surrogate_reward(ctx['source'],ctx['target'],rj['design'],A)}
        verdict_audit=audit_pair(rec['provenance'],sur['chosen'],sur['rejected']);audit_counts[rec['provenance']+':'+verdict_audit]+=1
        if verdict_audit=='misordered':drop(rec,'synthetic_misordered',surrogate=sur);continue
        if verdict_audit=='reward_disagreement':disagreements.append({'pref_id':rec['pref_id'],'provenance':rec['provenance'],'surrogate':sur})
        images=[_render(ctx['source'],out/'images')] if render else []
        if render and kind=='edit':images.append(_render(ctx['state'],out/'images'))
        task={'source':ctx['source'],'target':ctx['target'],'options':ctx.get('options') or {}}
        if kind=='edit':task['state']=ctx['state']
        row={'row_id':sha256_text('dpo|'+rec['pref_id']+'|'+ch['revision']+'|'+rj['revision'])[:20],'pref_id':rec['pref_id'],'kind':kind,
             'provenance':rec['provenance'],'verdict':rec['verdict'],'presentation':rec['presentation'],'chosen_id':orient[0],'rejected_id':orient[1],
             'source_family':family,'split':split,'split_manifest_sha256':mhash,'design_id':rec.get('design_id'),'source_hash':design_hash(ctx['source']),
             'target':ctx['target'],'locks':ctx.get('locks') or {},'analysis_options':ctx.get('options') or {},'context_hash':chash,'image_transform':IMAGE_TRANSFORM,
             'images':['../images/'+Path(i).name for i in images],'prompt':prompt_for(kind,ctx,A),'chosen':ch['completion'],'rejected':rj['completion'],
             'chosen_revision':ch['revision'],'rejected_revision':rj['revision'],'surrogate':sur,'audit':verdict_audit,
             'reward_disagreement':verdict_audit=='reward_disagreement','task':_dumps(task)}
        for k in ('teacher','review','event_id'):
            if k in rec:row[k]=rec[k]
        if kind=='plan':row['plan_schema']=ctx.get('plan_schema',PLAN_V2)
        else:row.update(action_schema=EDIT_V1,orientation='base_to_result',base_revision=design_hash(ctx['state']),state_defects=ctx.get('state_defects') or {})
        binary.append(row);counts['binary_'+kind]+=1
    (out/'nonbinary.jsonl').write_text(''.join(_dumps(r)+'\n' for r in nonbinary))
    (out/'excluded.jsonl').write_text(''.join(_dumps(r)+'\n' for r in excluded_rows))
    images={p.name:sha256_text(p.read_bytes()) for p in sorted((out/'images').glob('*.png'))}
    variant_manifests={}
    for name,allowed in variants.items():
        vdir=out/name;vdir.mkdir(exist_ok=True);files={};vcounts=Counter()
        vrows=[r for r in binary if r['provenance'] in allowed]
        for (kind,split),fname in FILES.items():
            sel=sorted((r for r in vrows if r['kind']==kind and r['split']==split),key=lambda r:r['row_id'])
            text=''.join(_dumps(r)+'\n' for r in sel);(vdir/fname).write_text(text);files[fname]=sha256_text(text);vcounts[f'{split}/{kind}']=len(sel)
        prov=Counter(r['provenance'] for r in vrows)
        m={'dataset_schema':DPO_DATASET_SCHEMA,'variant':name,'allowed_provenance':list(allowed),'plan_schema':PLAN_V2,'action_schema':EDIT_V1,
           'image_transform':IMAGE_TRANSFORM,'counts':dict(sorted(vcounts.items())),'provenance':dict(sorted(prov.items())),'human_rows':prov.get('explicit_human',0),
           'split_manifest_sha256':mhash,'files':files,'images':images,'length':{k:length_report([r for r in vrows if r['kind']==k]) for k in ('plan','edit')},
           'reward_disagreements':sum(r['reward_disagreement'] for r in vrows),
           'note':'binary decisive pairs only; tie/both_bad/unsure live in ../nonbinary.jsonl' + ('' if prov.get('explicit_human') else '; NO human-reviewed pairs: not evidence of designer preference')}
        (vdir/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');variant_manifests[name]={'counts':m['counts'],'provenance':m['provenance']}
    manifest={'dataset_schema':DPO_DATASET_SCHEMA+'+root','pref_schema':PREF_SCHEMA,'counts':dict(sorted(counts.items())),
              'audit':{'by_provenance':dict(sorted(audit_counts.items())),'reward_disagreements':disagreements,
                       'surrogate':'training.reward.reward_from(assess(...)) '+str(getattr(__import__('training.reward',fromlist=['x']),'REWARD_VERSION','unversioned (lane fixture; Codex final: preservation-rubric-v2)')),'policy':'synthetic/teacher: surrogate must agree or the pair is excluded; explicit_human/behavioral: disagreement kept and flagged'},
              'nonbinary':dict(Counter(r['verdict'] for r in nonbinary)),'variants':variant_manifests,'split_manifest_sha256':mhash,
              'inputs':{str(p):sha256_text(Path(p).read_bytes()) for p in pref_paths},'images':len(images)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+'\n')
    return manifest


# ----------------------------------------------------------------- validate

def is_dpo_dataset(data):
    m=Path(data)/'manifest.json'
    return m.exists() and json.loads(m.read_text()).get('dataset_schema')==DPO_DATASET_SCHEMA


def load_pairs(data,kind,split='train'):
    p=Path(data)/FILES[(kind,split)]
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def validate_pair_row(row,data=None,excl=None,split_manifest=None,audit=False):
    """Every reason this binary row cannot be trained on. Re-materializes both sides through the runtime path."""
    reasons=check_provenance(row)
    for k in ('row_id','kind','verdict','presentation','chosen_id','rejected_id','chosen','rejected','task','prompt','source_hash','split','source_family'):
        if k not in row:reasons.append('missing_'+k)
    if reasons:return reasons
    if row['verdict'] not in DECISIVE:return reasons+['non_binary_verdict_in_binary_rows']
    pres=row['presentation']
    if pres.get(row['verdict'])!=row['chosen_id'] or pres.get('B' if row['verdict']=='A' else 'A')!=row['rejected_id']:reasons.append('orientation_mismatch')
    task=json.loads(row['task']);kind=row['kind']
    ctx={'source':task['source'],'target':task['target'],'locks':row.get('locks') or {},'plan_schema':row.get('plan_schema',PLAN_V2),'options':task.get('options') or row.get('analysis_options') or {}}
    if kind=='edit':
        if 'state' not in task:return reasons+['missing_state']
        ctx['state']=task['state']
        if design_hash(task['state'])!=row.get('base_revision'):reasons.append('base_revision_mismatch')
        if row.get('orientation')!='base_to_result':reasons.append('bad_orientation')
    if design_hash(task['source'])!=row['source_hash']:reasons.append('source_hash_mismatch')
    if task['target']!=row.get('target'):reasons.append('target_mismatch')
    if context_hash(kind,ctx)!=row.get('context_hash'):reasons.append('context_hash_mismatch')
    if row['split'] not in EXPORT_SPLITS:reasons.append('non_exported_split')
    if split_manifest is not None:
        try:
            if split_of(row['source_family'],split_manifest)!=row['split']:reasons.append('split_mismatch')
        except ValueError:reasons.append('family_not_in_manifest')
    if excluded(excl or suite_exclusions(),source=task['source'],family=row['source_family'],design_id=row.get('design_id')):reasons.append('visual_suite_source')
    if data is not None:
        for im in row.get('images',[]):
            if not (Path(data)/im).is_file():reasons.append('missing_image')
    A=analyze(task['source'],group_policies=(ctx.get('options') or {}).get('group_policies'));designs={}
    for side in ('chosen','rejected'):
        obj=parse_plan_text(row[side])
        if obj is None or canonical_completion(kind,obj)!=row[side]:reasons.append(side+'_not_canonical');continue
        status,design,_=materialize_side(kind,ctx,row[side],A)
        if status!='ok':reasons.append(side+'_'+status);continue
        designs[side]=design
        if design_hash(design)!=row.get(side+'_revision'):reasons.append(side+'_revision_mismatch')
    if len(designs)==2:
        px,same=geometry_gap(designs['chosen'],designs['rejected'])
        if px<NEAR_PX and same:reasons.append('near_identical_sides')
        if audit:
            v=audit_pair(row['provenance'],surrogate_reward(task['source'],task['target'],designs['chosen'],A),surrogate_reward(task['source'],task['target'],designs['rejected'],A))
            if v=='misordered':reasons.append('synthetic_misordered')
            if v=='reward_disagreement' and not row.get('reward_disagreement'):reasons.append('unflagged_reward_disagreement')
    return reasons


def validate_dataset(data,audit=False):
    data=Path(data);excl=suite_exclusions();rep={'rows':0,'invalid':0,'reasons':Counter(),'provenance':Counter(),'kinds':Counter(),'reward_disagreements':0}
    root=data.parent if (data.parent/'split_manifest.json').exists() else data
    sm=json.loads((root/'split_manifest.json').read_text()) if (root/'split_manifest.json').exists() else None
    for (kind,split) in FILES:
        for row in load_pairs(data,kind,split):
            rep['rows']+=1;rep['provenance'][row.get('provenance')]+=1;rep['kinds'][f'{split}/{kind}']+=1;rep['reward_disagreements']+=bool(row.get('reward_disagreement'))
            rs=validate_pair_row(row,data,excl,sm,audit)
            if rs:rep['invalid']+=1;rep['reasons'].update(rs)
    rep={k:(dict(v) if isinstance(v,Counter) else v) for k,v in rep.items()}
    rep['ok']=rep['rows']>0 and rep['invalid']==0
    return rep


# ------------------------------------------------------ preference sources

def _presented(pref_id,chosen,rejected):
    """Seeded A/B randomization, exactly as a blinded review UI would show the pair."""
    if random.Random(pref_id).random()<.5:return {'A':chosen,'B':rejected},'A'
    return {'A':rejected,'B':chosen},'B'


def synthesize(demo_paths,out=None):
    """Engine demonstrations -> deterministic_synthetic preference records (no human signal).

    plan pairs: final engine design vs each weaker same-target candidate (both as layout-plan/2).
    edit pairs: same base state; chosen = edit to the final design, rejected = edit to another weaker candidate."""
    recs=[]
    for p in demo_paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():continue
            d=json.loads(line);src,tgt=d['source'],d['target'];final=d['final']
            teacher='engine.pipeline.generate rubric tier2 overall (L15 demonstrations)'
            base={'pref_schema':PREF_SCHEMA,'provenance':'deterministic_synthetic','teacher':teacher,'source_family':d['family'],'design_id':d['design_id']}
            for n,st in enumerate(d['states']):
                if st['overall'] is None or final['overall'] is None or st['overall']>=final['overall']:continue
                pid=sha256_text(f"plan|{d['demo_id']}|{n}")[:20];pres,v=_presented(pid,'final','state%d'%n)
                recs.append({**base,'pref_id':pid,'kind':'plan','verdict':v,'presentation':pres,'context':{'source':src,'target':tgt,'plan_schema':PLAN_V2},
                             'candidates':{'final':{'design':final['design']},'state%d'%n:{'design':st['design']}},
                             'teacher_scores':{'final':final['overall'],'state%d'%n:st['overall']}})
            for i,base_st in enumerate(d['states']):
                for j,other in enumerate(d['states']):
                    if i==j or other['overall'] is None or final['overall'] is None or other['overall']>=final['overall']:continue
                    pid=sha256_text(f"edit|{d['demo_id']}|{i}|{j}")[:20];pres,v=_presented(pid,'to_final','to_state%d'%j)
                    recs.append({**base,'pref_id':pid,'kind':'edit','verdict':v,'presentation':pres,
                                 'context':{'source':src,'target':tgt,'state':base_st['design'],'state_defects':base_st['defects']},
                                 'candidates':{'to_final':{'design':final['design']},'to_state%d'%j:{'design':other['design']}},
                                 'teacher_scores':{'to_final':final['overall'],'to_state%d'%j:other['overall']}})
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(''.join(_dumps(r)+'\n' for r in sorted(recs,key=lambda r:r['pref_id'])))
    return recs


def import_legacy(data,out=None):
    """training.export dpo rows (rubric pairs, layout-plan/1) -> deterministic_synthetic preference records."""
    recs=[]
    for split in ('train','validation'):
        p=Path(data)/('dpo.jsonl' if split=='train' else 'dpo.validation.jsonl')
        if not p.exists():continue
        for line in p.read_text().splitlines():
            if not line.strip():continue
            r=json.loads(line);task=json.loads(r['task'])
            pid=sha256_text('legacy|'+r['chosen']+'|'+r['rejected']+'|'+r['task'])[:20];pres,v=_presented(pid,'chosen','rejected')
            recs.append({'pref_schema':PREF_SCHEMA,'pref_id':pid,'kind':'plan','provenance':'deterministic_synthetic',
                         'teacher':'training.export rubric reward (margin %s)'%r.get('reward_chosen'),'source_family':str(r['family']),'design_id':str(r['family']),
                         'verdict':v,'presentation':pres,'context':{'source':task['source'],'target':task['target'],'plan_schema':PLAN_V1},
                         'candidates':{'chosen':{'completion':r['chosen']},'rejected':{'completion':r['rejected']}}})
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(''.join(_dumps(r)+'\n' for r in sorted(recs,key=lambda r:r['pref_id'])))
    return recs


# ------------------------------------------------------------------ trainer

DPO_SWEEP={
    'loss':'sigmoid DPO (Rafailov et al. 2023), sum of completion-token log-probs; no SFT anchor, no label smoothing (plain DPO first)',
    'reference':'frozen copy of the SFT LoRA adapter (peft adapter "ref"), active only under torch.no_grad + inference_mode',
    'policy_init':'the same SFT adapter, trainable ("default"); LoRA rank/targets inherited from the SFT bundle',
    'grid':[{'beta':b,'lr':lr} for b in (.05,.1,.2) for lr in (5e-6,1e-5,2e-5)],
    'lr_note':'all DPO learning rates sit below the SFT sweep (5e-5, 1e-4)',
    'dropout':'disabled (model.eval()) so policy == reference exactly at step 0 and reference scores are reproducible',
    'effective_batch':8,'per_pair_forward':'chosen and rejected encoded separately with training.vlm_adapters.encode_example (asserted masking)',
    'max_epochs':2,'early_stop':{'metric':'held-out validation preference accuracy (policy implicit-reward margin > 0)','tiebreak':'validation DPO loss','patience_evals':2},
    'monitor':['raw_valid_rate and parse_rate of greedy generations on held-out prompts (evaluate_policy)','diversity = unique valid materialized designs / generations',
               'chosen/rejected log-prob drift (both falling = likelihood displacement)','reference log-prob audit (must be bit-identical)'],
    'abort':'raw_valid_rate more than 5 points below the SFT bundle on the same prompts, or diversity collapse to one design per context',
    'variants':list(VARIANTS),
}


def attach_reference(model):
    """Add a frozen 'ref' LoRA adapter that is an exact copy of 'default' (the SFT policy). Same mechanism as
    TRL 1.16 DPOTrainer for a PeftModel with a pretrained adapter; NOT the disabled-adapter base model."""
    import torch
    model.add_adapter('ref',model.peft_config['default'])
    with torch.no_grad():
        for name,param in model.named_parameters():
            if '.default.' in name:model.get_parameter(name.replace('.default.','.ref.')).data.copy_(param.data)
    model.set_adapter('default')
    for name,param in model.named_parameters():
        if '.ref.' in name:param.requires_grad_(False)
    return model


def reference_fingerprint(model):
    import torch
    h=hashlib.sha256()
    for name,param in sorted(model.named_parameters()):
        if '.ref.' in name:h.update(name.encode());h.update(param.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def sequence_logp(model,ex,device,dtype):
    """Sum of log-probs of the supervised (completion + end-of-turn) tokens; only completion logits are materialized."""
    import torch
    from training.vlm_adapters import IGNORE,_to
    from training.vlm_adapters import collate
    batch=_to(collate([ex],0),device,dtype);labels=batch.pop('labels')[0]
    sup=(labels!=IGNORE).nonzero().flatten();first=int(sup[0]);L=labels.shape[0]
    out=model(**batch,logits_to_keep=L-first+1)
    logits=out.logits[0,:-1].float()                  # predicts tokens first .. L-1
    target=labels[first:];mask=target!=IGNORE
    lp=torch.log_softmax(logits,-1).gather(-1,target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp*mask).double().sum()  # float64 accumulation: sums are O(1e4) nats, margins O(1e-2)


def dpo_loss(pc,pr,rc,rr,beta):
    """Sigmoid DPO on summed log-probs; reference terms are float64 (python floats or tensors)."""
    import torch
    rc,rr=(torch.as_tensor(x,dtype=pc.dtype,device=pc.device) for x in (rc,rr))
    margin=(pc-rc)-(pr-rr)
    return -torch.nn.functional.logsigmoid(beta*margin),margin


class _PolicyView:
    """The training model seen through the runtime generate() path (policy adapter active)."""
    def __init__(self,model,processor,adapter,device):self.model,self.processor,self.adapter,self.device=model,processor,adapter,device
    def generate(self,images,prompt,max_new_tokens=768):
        from engine.vlm import TransformersPolicy
        return TransformersPolicy.generate(self,images,prompt,max_new_tokens)


def _encode_pairs(rows,data,processor):
    from training.vlm_adapters import _load_images,assert_supervision,encode_example
    out=[]
    for r in rows:
        images=_load_images(r,data)
        c=encode_example(processor,images,r['prompt'],r['chosen']);assert_supervision(processor,c)
        j=encode_example(processor,images,r['prompt'],r['rejected']);assert_supervision(processor,j)
        out.append((c,j))
    return out


def load_sft_policy(sft_bundle,device='cpu',dtype=None):
    """SFT bundle -> (PeftModel with trainable 'default' + frozen 'ref', processor, adapter, manifest)."""
    import torch,transformers
    from peft import PeftModel
    from engine.vlm import ADAPTERS,check_bundle
    chk=check_bundle(sft_bundle,verify_hashes=True)
    if chk['status']!='ok':raise RuntimeError('SFT bundle '+chk['status']+': '+'; '.join(chk['reasons']))
    m=chk['manifest'];a=ADAPTERS[m['adapter_id']];base=m['base'];dtype=dtype or m.get('dtype','float32')
    processor=transformers.AutoProcessor.from_pretrained(str(Path(sft_bundle)/'processor'),local_files_only=True)
    cls=getattr(transformers,a.model_class,None) or transformers.AutoModelForImageTextToText
    model=cls.from_pretrained(base.get('local_path') or base['repo'],revision=None if base.get('local_path') else base.get('revision'),dtype=getattr(torch,dtype),local_files_only=True)
    model=PeftModel.from_pretrained(model,str(Path(sft_bundle)/'adapter'),is_trainable=True).to(device)
    attach_reference(model);model.eval()
    return model,processor,a,m


def train_dpo(train_rows,val_rows,data,sft_bundle,out,*,beta=.1,lr=1e-5,max_steps=20,epochs=1,accum=1,eval_every=2,patience=2,
              device='cpu',seed=7,dataset_manifest=None,monitor_generations=2,max_new_tokens=64):
    """Explicit DPO loop over L15's asserted-masking encoder. Returns a typed result and exports a bundle."""
    import torch
    from training.vlm_adapters import evaluate_policy,export_bundle,gradient_report,peak_rss_gb,select_device,_strip
    from engine.vlm import check_bundle
    torch.manual_seed(seed);device=select_device(device);out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if not train_rows:return {'status':'no_training_rows'}
    if device=='mps':return {'status':'unsupported_device','reason':'DPO summed-logprob float64 reference audit requires CPU or CUDA; MPS float64 unsupported'}
    if isinstance(accum,bool) or accum<1:raise ValueError('accum must be positive')
    model,processor,a,sft_manifest=load_sft_policy(sft_bundle,device)
    dtype=sft_manifest.get('dtype','float32')
    train_ex=_encode_pairs(train_rows,data,processor);val_ex=_encode_pairs(val_rows,data,processor)
    params=[p for n,p in model.named_parameters() if p.requires_grad]
    if any('.default.' not in n for n,p in model.named_parameters() if p.requires_grad):return {'status':'bad_trainable_set'}
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.0)

    def ref_scores(pairs):
        model.set_adapter('ref',inference_mode=True) if _accepts_inference_mode(model) else model.set_adapter('ref')
        try:
            with torch.no_grad():return [(float(sequence_logp(model,c,device,dtype)),float(sequence_logp(model,j,device,dtype))) for c,j in pairs]
        finally:
            model.set_adapter('default')
            for n,p in model.named_parameters():
                if '.ref.' in n:p.requires_grad_(False)
    ref_train=ref_scores(train_ex);ref_val=ref_scores(val_ex)
    fp0=reference_fingerprint(model)

    def evaluate():
        tot=0.0;acc=0;margins=[];pcs=[];prs=[]
        with torch.no_grad():
            for (c,j),(rc,rr) in zip(val_ex,ref_val):
                pc=sequence_logp(model,c,device,dtype);pr=sequence_logp(model,j,device,dtype)
                loss,m=dpo_loss(pc,pr,rc,rr,beta);tot+=float(loss);acc+=float(m)>0;margins.append(float(m));pcs.append(float(pc));prs.append(float(pr))
        n=len(val_ex)
        return {'val_loss':round(tot/n,6),'val_pref_acc':round(acc/n,4),'val_margin_mean':round(statistics.fmean(margins),6),
                'val_logp_chosen':round(statistics.fmean(pcs),4),'val_logp_rejected':round(statistics.fmean(prs),4)} if n else {}

    def monitor():
        if not monitor_generations or not val_rows:return None
        rows=[{**r,'completion':r['chosen']} for r in val_rows[:monitor_generations]]
        rep=evaluate_policy(_PolicyView(model,processor,a,device),rows,data,None,max_new_tokens)
        valid=[x for x in rep['records'] if x.get('raw_valid')]
        # crude diversity proxy: distinct 80-char reply prefixes among raw-valid generations (None when nothing is valid)
        rep['distinct_valid_reply_prefixes']=round(len({x.get('reply_head') for x in valid})/len(valid),4) if valid else None
        return {k:rep[k] for k in ('tasks','parse_rate','raw_valid_rate','repaired_valid_rate','reasons','distinct_valid_reply_prefixes')}

    log=(out/'dpo_log.jsonl').open('w');history=[];grad=None;status='trained';step=0;bad=0
    initial=evaluate();initial['step']=0;initial['monitor']=monitor();history.append(initial);log.write(json.dumps(initial)+'\n')
    best={'key':(initial.get('val_pref_acc',0),-initial.get('val_loss',0)),'step':0,'state':_adapter_state(model)}
    rng=random.Random(seed);order=list(range(len(train_ex)))
    for epoch in range(epochs):
        rng.shuffle(order)
        for n,i in enumerate(order):
            c,j=train_ex[i];rc,rr=ref_train[i]
            pc=sequence_logp(model,c,device,dtype);pr=sequence_logp(model,j,device,dtype)
            loss,margin=dpo_loss(pc,pr,rc,rr,beta)
            if not torch.isfinite(loss):status='non_finite_loss';break
            (loss/accum).backward()
            if (n+1)%accum and n+1!=len(order):continue
            pending=(n%accum)+1
            if pending<accum:
                for param in params:
                    if param.grad is not None:param.grad.mul_(accum/pending)
            if grad is None:
                grad=gradient_report(model);grad['grad_ref']=[nm for nm,p in model.named_parameters() if '.ref.' in nm and p.grad is not None][:5]
                grad['grad_outside_policy']=[nm for nm,p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0 and '.default.' not in nm][:5]
                if grad['grad_non_lora'] or grad['trainable_non_lora'] or grad['grad_ref'] or grad['grad_outside_policy'] or not grad['grad_finite'] or grad['with_grad']==0:status='bad_gradients';break
            opt.step();opt.zero_grad();step+=1
            rec={'step':step,'epoch':epoch,'loss':round(float(loss),6),'margin':round(float(margin),6),'logp_chosen':round(float(pc),4),'logp_rejected':round(float(pr),4)}
            if val_ex and step%eval_every==0:
                rec.update(evaluate());rec['monitor']=monitor()
                key=(rec['val_pref_acc'],-rec['val_loss'])
                if key>best['key']:best={'key':key,'step':step,'state':_adapter_state(model)};bad=0
                else:bad+=1
            history.append(rec);log.write(json.dumps(rec)+'\n');log.flush()
            if step>=max_steps or bad>=patience:break
        if status!='trained' or step>=max_steps or bad>=patience:break
    log.close()
    # Reference audit: frozen weights and bit-identical reference scores after training.
    ref_after=ref_scores(train_ex[:1]+val_ex)
    ref_before=ref_train[:1]+ref_val
    ref_audit={'fingerprint_before':fp0,'fingerprint_after':reference_fingerprint(model),
               'max_abs_logp_change':max((abs(x-y) for b,c in zip(ref_before,ref_after) for x,y in zip(b,c)),default=0.0)}
    ref_audit['fixed']=ref_audit['fingerprint_before']==ref_audit['fingerprint_after'] and ref_audit['max_abs_logp_change']==0.0
    if not ref_audit['fixed'] and status=='trained':status='reference_moved'
    if step==0 and status=='trained':status='no_optimizer_steps'
    config={'stage':'dpo','sft_bundle':str(sft_bundle),'sft_bundle_manifest_sha256':sha256_text(_dumps(sft_manifest)),'adapter_id':a.adapter_id,
            'beta':beta,'lr':lr,'max_steps':max_steps,'epochs':epochs,'accum':accum,'seed':seed,'device':device,'dtype':dtype,'steps_done':step,
            'best_step':best['step'],'early_stop':bad>=patience,'reference':DPO_SWEEP['reference'],'loss':'sigmoid',
            'train_pairs':len(train_rows),'val_pairs':len(val_rows),'pair_ids':sorted(r['row_id'] for r in train_rows+val_rows),
            'provenance':dict(Counter(r['provenance'] for r in train_rows+val_rows))}
    result={'status':status,'steps':step,'history':history,'gradients':grad,'reference_audit':ref_audit,'best_step':best['step'],
            'seconds':round(time.monotonic()-started,2),'peak_rss_gb':peak_rss_gb(),'config':config}
    if status!='trained':return result
    _load_adapter_state(model,best['state']);model.delete_adapter('ref')
    plan_schema=(dataset_manifest or {}).get('plan_schema',PLAN_V2);kinds=sorted({r['kind'] for r in train_rows})
    base=sft_manifest['base'].get('local_path')
    export_bundle(out/'bundle',model,processor,a,plan_schema,tasks=kinds,base=base,dtype=dtype,dataset_manifest=dataset_manifest,train_config=config,
                  metrics={'best':{k:v for k,v in history[[h['step'] for h in history].index(best['step'])].items() if k!='monitor'},'initial':{k:v for k,v in initial.items() if k!='monitor'}},
                  status='trained_dpo_'+('smoke' if a.verification=='ci_fixture' else 'candidate'))
    result['bundle']=str(out/'bundle');result['bundle_check']=check_bundle(out/'bundle',verify_hashes=True)['status']
    return result


def _accepts_inference_mode(model):
    import inspect
    try:return 'inference_mode' in inspect.signature(model.set_adapter).parameters
    except (TypeError,ValueError):return False


def _adapter_state(model):
    return {n:p.detach().clone() for n,p in model.named_parameters() if '.default.' in n}


def _load_adapter_state(model,state):
    import torch
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in state:p.data.copy_(state[n])


def smoke(data,sft_bundle,out,device='cpu',max_steps=4,beta=.1,lr=1e-5,n_train=4,n_val=2):
    """Real few-step DPO on the cached tiny model, export, reload through engine.vlm, generate, parse-or-reject."""
    from training.vlm_adapters import evaluate_policy,select_device,_strip
    from engine.vlm import TransformersPolicy
    manifest=json.loads((Path(data)/'manifest.json').read_text())
    train=(load_pairs(data,'plan','train')[:n_train//2]+load_pairs(data,'edit','train')[:n_train-n_train//2])
    val=(load_pairs(data,'plan','validation')+load_pairs(data,'edit','validation'))[:n_val] or (load_pairs(data,'plan','train')[n_train//2:]+load_pairs(data,'edit','train')[n_train:])[:n_val]
    res=train_dpo(train,val,data,sft_bundle,out,beta=beta,lr=lr,max_steps=max_steps,eval_every=2,device=device,dataset_manifest=manifest)
    if res.get('bundle'):
        pol=TransformersPolicy.load(res['bundle'],select_device(device))
        rows=[{**r,'completion':r['chosen']} for r in val+train[:1]]
        res['reload_generation']=_strip(evaluate_policy(pol,rows,data,None,64))
    (Path(out)/'smoke_result.json').write_text(json.dumps(res,indent=2,default=str)+'\n')
    return res


EXPERIMENT_SPEC={
    'experiment':'chameleon-dpo-vs-sft/1 (not executed: prerequisites below)',
    'prerequisites':['explicit_human pairs: chameleon-pref/1 records with review{reviewer_id, committed_at, verdict}, blinded A/B order recorded, source+target+base revision on both sides',
                     'pilot size >= 500 decisive human pairs across >= 60 non-suite families (re-size after a repeat-label disagreement estimate); ties/both_bad stored, not trained',
                     'a promoted SFT bundle (guide 15) on a real VLM (qwen2.5-vl-3b-instruct first) with pinned revision',
                     'candidate pool: >= 4 SFT samples (temperature 0.7) + 2 deterministic engine alternatives per context, all structurally valid before review',
                     'Codex workspace export with reviewer_id/presentation order/revision hashes (see handoff requests)'],
    'data':{'split':'frozen family split manifest shared with SFT (training.families); visual-suite families excluded','variants':['human_only','human_plus_synthetic'],
            'synthetic':'engine-demonstration pairs (deterministic_synthetic) only in human_plus_synthetic; surrogate-ordered'},
    'reference':'frozen SFT adapter copy (exactly the promoted SFT bundle); reference log-probs recomputed per step (TRL 1.16 refuses precompute for vision datasets)',
    'sweep':DPO_SWEEP['grid'],'selection':'held-out valid-preference accuracy, then validation DPO loss; reject runs whose raw validity falls > 5 points below SFT',
    'comparison':{'arms':['sft (rollback artifact)','dpo:human_only','dpo:human_plus_synthetic'],
                  'equal_budget':'same prompts, sample count (k=4), greedy + T=0.7 decoding, max_new_tokens, solver/repair budget, end-to-end pipeline resource profile',
                  'metrics':['blind designer pairwise preference (family-clustered win/tie/loss, bootstrap CI)','raw_valid_rate','repaired_valid_rate',
                             'diversity (unique valid designs per context)','latency median/p95','peak memory','surrogate reward (auxiliary only)']},
    'acceptance':'promote DPO only if family-clustered designer preference improves over SFT (CI excludes 0) with no material raw-validity, diversity or cost regression and no new structural/blocking failure in the 16-case suite; a surrogate-reward or margin gain alone never promotes',
    'rollback':'unset CHAMELEON_VLM_BUNDLE or point it back at the exact SFT bundle (hash recorded in the DPO bundle training.sft_bundle_manifest_sha256)',
}


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('synthesize');s.add_argument('demos',type=Path,nargs='+');s.add_argument('--out',type=Path,required=True)
    l=sub.add_parser('import-legacy');l.add_argument('data',type=Path);l.add_argument('--out',type=Path,required=True)
    b=sub.add_parser('build');b.add_argument('prefs',type=Path,nargs='+');b.add_argument('--out',type=Path,required=True);b.add_argument('--split-manifest',type=Path)
    b.add_argument('--no-render',action='store_true');b.add_argument('--no-audit',action='store_true')
    v=sub.add_parser('validate');v.add_argument('data',type=Path);v.add_argument('--audit',action='store_true')
    t=sub.add_parser('train');t.add_argument('data',type=Path);t.add_argument('--sft-bundle',type=Path,required=True);t.add_argument('--out',type=Path,required=True)
    t.add_argument('--beta',type=float,default=.1);t.add_argument('--lr',type=float,default=1e-5);t.add_argument('--max-steps',type=int,default=20);t.add_argument('--epochs',type=int,default=1)
    t.add_argument('--accum',type=int,default=1);t.add_argument('--device',default='cpu');t.add_argument('--smoke',action='store_true',help='few pairs + reload/generate check')
    sub.add_parser('sweep-spec');sub.add_parser('experiment-spec')
    a=ap.parse_args()
    if a.cmd=='synthesize':r=synthesize(a.demos,a.out);print(json.dumps({'records':len(r),'kinds':dict(Counter(x['kind'] for x in r)),'verdicts':dict(Counter(x['verdict'] for x in r))}))
    elif a.cmd=='import-legacy':r=import_legacy(a.data,a.out);print(json.dumps({'records':len(r)}))
    elif a.cmd=='build':
        sm=json.loads(a.split_manifest.read_text()) if a.split_manifest else None
        print(json.dumps(build(a.prefs,a.out,sm,not a.no_render,not a.no_audit),indent=2,default=str))
    elif a.cmd=='validate':r=validate_dataset(a.data,a.audit);print(json.dumps(r,indent=2));raise SystemExit(0 if r['ok'] else 1)
    elif a.cmd=='train':
        man=json.loads((a.data/'manifest.json').read_text())
        if a.smoke:r=smoke(a.data,a.sft_bundle,a.out,a.device,a.max_steps,a.beta,a.lr)
        else:
            tr=load_pairs(a.data,'plan','train')+load_pairs(a.data,'edit','train');va=load_pairs(a.data,'plan','validation')+load_pairs(a.data,'edit','validation')
            r=train_dpo(tr,va,a.data,a.sft_bundle,a.out,beta=a.beta,lr=a.lr,max_steps=a.max_steps,epochs=a.epochs,accum=a.accum,device=a.device,dataset_manifest=man)
        print(json.dumps({k:v for k,v in r.items() if k!='history'},indent=2,default=str));raise SystemExit(0 if r['status']=='trained' else 1)
    elif a.cmd=='sweep-spec':print(json.dumps(DPO_SWEEP,indent=2))
    else:print(json.dumps(EXPERIMENT_SPEC,indent=2))
