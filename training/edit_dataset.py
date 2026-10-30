"""Whole-plan and edit-imitation SFT datasets with mandatory provenance (guide 15.4).

  # 1. expensive, non-deterministic (refinement is time-budgeted): run the engine on non-suite sources
  python -m training.edit_dataset collect output/crello/designs --out work/learn/demos.jsonl --limit 8 --targets 1080x1080,1584x396
  # 2. deterministic, idempotent: demonstrations -> rows (+ images, split manifest, dataset manifest)
  python -m training.edit_dataset build work/learn/demos.jsonl --out work/learn/data
  # 3. re-materialize and validate every row through the runtime path
  python -m training.edit_dataset validate work/learn/data

Row kinds (one JSON object per line, sorted by row_id):

* ``plan``: source image + plan prompt -> layout-plan completion (v2 by default; v1 rows are
  admitted only when their materialization is not lossy).
* ``edit``: state (current design + its defects, source image + state image) -> one
  ``edit-action/1`` completion. Orientation is always ``base_revision -> result_revision``;
  the action becomes an ``engine.contracts.EditProposal`` and is materialized by
  ``engine.materialize.apply_edit`` (exact runtime path).

Provenance is one of ``explicit_human | behavioral | teacher_model | deterministic_synthetic``
and is never inferred from success. Engine demonstrations are ``deterministic_synthetic``.
The four visual-suite families are excluded at collection and again at build/validation.
Only train/validation families are exported (calibration/test never).
"""
import argparse,hashlib,json,subprocess
from collections import Counter
from pathlib import Path
from engine.analysis import analyze
from engine.canonical import ASSETS_DIR,MUTABLE_FIELDS,ROOT
from engine.contracts import DesignRevision,design_hash
from engine.neural_proposals import (PLAN_V1,PLAN_V2,EDIT_V1,PlanRejected,action_to_proposal,arrangement_of,design_to_plan,
                                     design_to_plan_v2,edit_action,edit_prompt,materialize_plan,parse_plan_text,plan_fidelity,
                                     plan_prompt,plan_prompt_v2)

DATASET_SCHEMA='chameleon-learn-dataset/1'
PROVENANCE=('explicit_human','behavioral','teacher_model','deterministic_synthetic')
EXPORT_SPLITS=('train','validation')
SUITE_MANIFEST=ROOT/'docs/individual-update-plans/visual-suite.v1.json'
IMAGE_TRANSFORM='letterbox_white_672'
FILES={('plan','train'):'plan_sft.jsonl',('plan','validation'):'plan_sft.validation.jsonl',
       ('edit','train'):'edit_sft.jsonl',('edit','validation'):'edit_sft.validation.jsonl'}


def _dumps(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)


def sha256_text(text):
    return hashlib.sha256(text.encode() if isinstance(text,str) else text).hexdigest()


# ----------------------------------------------------- families and exclusion

def immutable_signature(source):
    values=[{k:v for k,v in e.items() if k not in MUTABLE_FIELDS and k!='id'} for e in source.get('elements',[])]
    return sha256_text(json.dumps(sorted(values,key=lambda x:json.dumps(x,sort_keys=True)),sort_keys=True))


def resolve_family(record,fallback=None):
    """Same precedence as Codex's training.families.resolve_family (used when available)."""
    from training import families
    if hasattr(families,'resolve_family'):return families.resolve_family(record,fallback)
    family=record.get('source_family') or record.get('family') or (record.get('metadata') or {}).get('source_id') or fallback
    if family is None:raise ValueError('source family provenance required')
    return str(family)


def build_split_manifest(records):
    """Codex's frozen family manifest builder when published; otherwise an exact compat copy of it
    (provenance + immutable-content union; sha256 70/10/5/15 train/validation/calibration/test)."""
    from training import families
    if hasattr(families,'build_split_manifest'):return families.build_split_manifest(records)
    parents={};fingerprints={};aliases={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for fallback,record in records:
        family=resolve_family(record,fallback);find(family);source=record.get('source',record)
        if source.get('elements'):
            sig=immutable_signature(source)
            if sig in fingerprints:union(family,fingerprints[sig])
            else:fingerprints[sig]=family
        aliases[str(fallback)]=family
    aliases={k:find(v) for k,v in aliases.items() if k!=find(v)}
    aliases.update({k:find(k) for k in parents if k!=find(k)})
    out={}
    for k in parents:
        root=find(k);bucket=int(hashlib.sha256(root.encode()).hexdigest()[:8],16)%100
        out[root]='train' if bucket<70 else 'validation' if bucket<80 else 'calibration' if bucket<85 else 'test'
    return {'schema_version':1,'families':out,'aliases':aliases,'excluded_families':[],
            'policy':'provenance + exact immutable content union; sha256 cluster 70/10/5/15 train/validation/calibration/test; reviewed near-duplicate aliases required',
            'builder':'training.families.build_split_manifest' if hasattr(families,'build_split_manifest') else 'training.edit_dataset compat copy of Codex families.build_split_manifest'}


def split_of(family,manifest):
    from training import families
    fn=getattr(families,'split_of',None)
    if fn is not None:
        try:return fn(family,manifest)
        except TypeError:pass
    family=str(family);aliases=manifest.get('aliases',{});seen=set()
    while family in aliases:
        if family in seen:raise ValueError('cyclic family alias')
        seen.add(family);family=str(aliases[family])
    if family in manifest.get('excluded_families',[]):raise ValueError('family excluded from training: '+family)
    split=manifest.get('families',{}).get(family)
    if split not in ('train','validation','calibration','test'):raise ValueError('family missing from frozen split manifest: '+family)
    return split


def suite_exclusions(path=SUITE_MANIFEST):
    """Identities of the frozen visual-suite sources: never training data."""
    suite=json.loads(Path(path).read_text());out={'design_ids':set(),'families':set(),'file_sha256':set(),'source_hashes':set(),'signatures':set()}
    for s in suite['sources']:
        out['file_sha256'].add(s['sha256'])
        if s.get('catalog_id'):out['design_ids'].add(s['catalog_id'])
        if s.get('source_id'):out['families'].add(s['source_id'])
        out['design_ids'].add(Path(s['path']).stem);out['families'].add(s['id'])
        f=ROOT/s['path']
        if f.is_file():
            record=json.loads(f.read_text());src=record.get(s.get('source_field','source'),record)
            out['source_hashes'].add(design_hash(src));out['signatures'].add(immutable_signature(src))
            fam=(record.get('metadata') or {}).get('source_id')
            if fam:out['families'].add(str(fam))
    return out


def excluded(excl,*,source,family,design_id=None,file_bytes=None):
    """Reason string when a source belongs to the visual suite (or is a content copy of one)."""
    if design_id and design_id in excl['design_ids']:return 'suite_design_id'
    if family in excl['families']:return 'suite_family'
    if file_bytes is not None and sha256_text(file_bytes) in excl['file_sha256']:return 'suite_file'
    if design_hash(source) in excl['source_hashes']:return 'suite_source_hash'
    if source.get('elements') and immutable_signature(source) in excl['signatures']:return 'suite_content_signature'
    return None


# ------------------------------------------------------------------ collect

def code_revision():
    def git(*a):
        try:return subprocess.run(['git','-C',str(ROOT),*a],capture_output=True,text=True,timeout=10).stdout.strip()
        except (OSError,subprocess.SubprocessError):return ''
    h=hashlib.sha256()
    for f in sorted((ROOT/'engine').rglob('*.py')):h.update(f.relative_to(ROOT).as_posix().encode());h.update(f.read_bytes())
    return {'git_head':git('rev-parse','HEAD') or None,'git_dirty':bool(git('status','--porcelain','--','engine','training')),'engine_tree_sha256':h.hexdigest()}


def _defects(candidate):
    q=(candidate.get('quality') or {}).get('defects') or {}
    out={k:sorted(map(str,v)) for k,v in sorted(q.items()) if v}
    t2=((candidate.get('rubric') or {}).get('tier2') or {})
    rub=sorted({str(d.get('id')) for d in t2.get('defects',[]) if isinstance(d,dict)})
    if rub:out['rubric']=rub
    return out


def _overall(candidate):
    t2=(candidate.get('rubric') or {}).get('tier2')
    return t2['overall'] if t2 else None


def collect(corpus,out,targets,limit=None,refine_seconds=4.0,min_overall=80.0,max_states=2,skip_files=()):
    """Run the deterministic engine on non-suite sources; one demonstration per (source, target)."""
    from engine.pipeline import generate
    from training.families import refuse_qa_corpus
    refuse_qa_corpus(corpus)
    excl=suite_exclusions();rev=code_revision();counts=Counter();out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    options={'render':False,'refine_seconds':float(refine_seconds)}
    paths=[p for p in sorted(Path(corpus).glob('*.json')) if p.stem not in set(skip_files)]
    with out.open('w') as fh:
        used=0
        for path in paths:
            if limit is not None and used>=limit:break
            raw=path.read_bytes()
            try:record=json.loads(raw);source=record.get('source',record)
            except ValueError:counts['unreadable']+=1;continue
            if not source.get('elements'):counts['empty']+=1;continue
            family=resolve_family(record,path.stem);meta=record.get('metadata') or {}
            why=excluded(excl,source=source,family=family,design_id=meta.get('id') or path.stem,file_bytes=raw)
            if why:counts['excluded_'+why]+=1;continue
            if meta.get('assets_available') is False:counts['missing_assets']+=1;continue
            used+=1
            for w,h in targets:
                target={'width':int(w),'height':int(h)}
                if source['canvas']==target:continue
                results=generate(source,target,options=options,all_candidates=True)
                ok=[r for r in results if r['valid'] and r['rubric']['gate']]
                if not ok:counts['no_valid_candidate']+=1;continue
                final=ok[0]
                if (_overall(final) or 0)<min_overall or not final['quality']['passed']:counts['final_below_bar']+=1;continue
                fh_=design_hash(final['design']);states=[];seen={fh_}
                seed=final['template'].replace('refined_','',1) if final['template'].startswith('refined_') else None
                pool=[r for r in results if r['valid'] and r['template']==seed]+[r for r in ok[1:] if (_overall(r) or 0)<(_overall(final) or 0)]
                for r in pool:
                    h_=design_hash(r['design'])
                    if h_ in seen or len(states)>=max_states:continue
                    seen.add(h_)
                    states.append({'design':r['design'],'template':r['template'],'overall':_overall(r),'defects':_defects(r),
                                   'relation':'refine_seed' if r['template']==seed else 'weaker_candidate'})
                demo={'demo_id':sha256_text(design_hash(source)+_dumps(target))[:16],'source_file':path.name,'design_id':meta.get('id') or path.stem,
                      'family':family,'source':source,'target':target,'provenance':'deterministic_synthetic','teacher':'engine.pipeline.generate',
                      'teacher_options':options,'code_revision':rev,'dataset':{'name':'cyberagent/crello' if 'crello' in str(corpus) else str(corpus),'license_note':'see output/crello/catalog.json'},
                      'final':{'design':final['design'],'template':final['template'],'overall':_overall(final),'defects':_defects(final)},'states':states}
                fh.write(_dumps(demo)+'\n');counts['demos']+=1;counts['states']+=len(states)
            print(path.stem,dict(counts),flush=True)
    return dict(counts)


# -------------------------------------------------------------------- build

def _render(design,folder):
    from engine.quality import resource_fingerprint,verify_design
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    name=sha256_text(design_hash(design)+resource_fingerprint(design))[:24]+'.png';path=folder/name
    if not path.exists():
        result=verify_design(design,output_path=path,max_size=672)
        if not result['verified']:raise ValueError('Training render unavailable: '+result.get('error','unknown'))
    return 'images/'+name


def check_provenance(row):
    p=row.get('provenance')
    if p not in PROVENANCE:return ['unknown_provenance']
    if p=='explicit_human':
        rv=row.get('review') or {}
        if not (rv.get('reviewer_id') and rv.get('committed_at') and rv.get('verdict')):return ['human_without_review_record']
    if p=='behavioral' and not row.get('event_id'):return ['behavioral_without_event']
    if p in ('teacher_model','deterministic_synthetic') and not row.get('teacher'):return ['missing_teacher']
    if p!='explicit_human' and row.get('review'):return ['review_on_non_human_row']
    return []


def _common(demo,split,kind,manifest_hash):
    return {'kind':kind,'provenance':demo['provenance'],'teacher':demo.get('teacher'),'review':demo.get('review'),'event_id':demo.get('event_id'),'source_family':demo['family'],'split':split,
            'split_manifest_sha256':manifest_hash,'design_id':demo['design_id'],'source_hash':design_hash(demo['source']),
            'target':demo['target'],'locks':demo.get('locks') or {},'analysis_options':{'group_policies':(demo.get('teacher_options') or {}).get('group_policies')},'code_revision':demo['code_revision'],'versions':__import__('training.reward',fromlist=['version_identity']).version_identity(),'image_transform':IMAGE_TRANSFORM}


def build(demo_paths,out,split_manifest=None,plan_schema=PLAN_V2,render=True):
    """Deterministic: same demonstrations (+ same manifest) -> byte-identical rows and manifest."""
    out=Path(out);(out/'images').mkdir(parents=True,exist_ok=True);excl=suite_exclusions();counts=Counter();fidelity=Counter()
    demos=[json.loads(l) for p in demo_paths for l in Path(p).read_text().splitlines() if l.strip()]
    demos.sort(key=lambda d:d['demo_id'])
    if split_manifest is None:
        split_manifest=build_split_manifest([(d['design_id'],{'family':d['family'],'source':d['source']}) for d in demos])
    mtext=_dumps(split_manifest);mhash=sha256_text(mtext);(out/'split_manifest.json').write_text(mtext+'\n')
    rows={k:[] for k in FILES}
    for d in demos:
        if check_provenance({**d,'teacher':d.get('teacher')}):counts['bad_provenance']+=1;continue
        why=excluded(excl,source=d['source'],family=d['family'],design_id=d['design_id'])
        if why:counts['excluded_'+why]+=1;continue
        split=split_of(d['family'],split_manifest)
        if split not in EXPORT_SPLITS:counts['skipped_'+split]+=1;continue
        source,target=d['source'],d['target'];A=analyze(source,group_policies=(d.get('teacher_options') or {}).get('group_policies'));final=d['final']['design']
        try:image=_render(source,out/'images') if render else None
        except ValueError:counts['source_render_unavailable']+=1;continue
        # Whole-plan row: completion must re-materialize to (a near-exact copy of) the engine's final.
        plan=design_to_plan_v2(final,source,A,arrangement_of(d['final']['template'])) if plan_schema==PLAN_V2 else design_to_plan(final,A,arrangement_of(d['final']['template']))
        outcome=materialize_plan(source,target,A,plan,d.get('locks'))
        if outcome.status!='ok' or not outcome.raw_valid:counts['plan_'+(outcome.status if outcome.status!='ok' else 'raw_invalid')]+=1
        else:
            fid=plan_fidelity(final,outcome.design);fidelity[plan_schema+':'+fid['class']]+=1
            if fid['class']=='lossy':counts['plan_lossy_excluded']+=1
            else:
                prompt=plan_prompt_v2(source,target,A,locks=d.get('locks')) if plan_schema==PLAN_V2 else plan_prompt(source,target,A)
                row={**_common(d,split,'plan',mhash),'row_id':sha256_text('plan|'+d['demo_id'])[:20],'plan_schema':plan_schema,
                     'images':[image] if image else [],'prompt':prompt,'completion':_dumps(plan),
                     'reference_revision':design_hash(final),'target_revision':design_hash(outcome.design),'fidelity':fid,
                     'raw_valid':outcome.raw_valid,'teacher_template':d['final']['template'],'teacher_overall':d['final']['overall'],
                     'task':_dumps({'source':source,'target':target,'options':d.get('teacher_options') or {}})}
                rows[('plan',split)].append(row)
        # Edit-imitation rows: state -> one typed edit -> engine final, through apply_edit.
        for n,st in enumerate(d['states']):
            from engine.materialize import apply_edit
            state=st['design'];revision=DesignRevision.create(state,origin='deterministic')
            action=edit_action(state,final,'imitate_'+st['relation'])
            if not action['patch']:counts['edit_noop']+=1;continue
            try:
                proposal=action_to_proposal(action,revision,provenance=d['provenance'],locks=d.get('locks'))
                result=apply_edit(source,revision,proposal,d.get('locks'))
            except (PlanRejected,ValueError) as exc:counts['edit_rejected']+=1;continue
            fid=plan_fidelity(final,result)
            if fid['class']=='lossy':counts['edit_lossy_excluded']+=1;continue
            images=[image,_render(state,out/'images')] if render else []
            row={**_common(d,split,'edit',mhash),'row_id':sha256_text(f'edit|{d["demo_id"]}|{n}|{revision.revision_hash}')[:20],'action_schema':EDIT_V1,
                 'images':images,'prompt':edit_prompt(source,target,A,state,st['defects'],locks=d.get('locks')),'completion':_dumps(action),
                 'orientation':'base_to_result','base_revision':revision.revision_hash,'result_revision':design_hash(result),
                 'reference_revision':design_hash(final),'fidelity':fid,'state_template':st['template'],'state_overall':st['overall'],
                 'state_defects':st['defects'],'result_overall':d['final']['overall'],'relation':st['relation'],
                 'task':_dumps({'source':source,'target':target,'state':state,'options':d.get('teacher_options') or {}})}
            rows[('edit',split)].append(row)
    files={}
    for key,name in FILES.items():
        text=''.join(_dumps(r)+'\n' for r in sorted(rows[key],key=lambda r:r['row_id']))
        (out/name).write_text(text);files[name]=sha256_text(text);counts[f'{key[1]}/{key[0]}']=len(rows[key])
    images={p.name:sha256_text(p.read_bytes()) for p in sorted((out/'images').glob('*.png'))}
    prov=Counter(r['provenance'] for v in rows.values() for r in v)
    manifest={'dataset_schema':DATASET_SCHEMA,'plan_schema':plan_schema,'action_schema':EDIT_V1,'image_transform':IMAGE_TRANSFORM,
              'counts':dict(sorted(counts.items())),'fidelity':dict(sorted(fidelity.items())),'provenance':dict(sorted(prov.items())),
              'split_manifest_sha256':mhash,'exported_splits':list(EXPORT_SPLITS),'files':files,'images':images,
              'demonstrations':{str(p):sha256_text(Path(p).read_bytes()) for p in demo_paths},
              'suite_excluded':sorted(excl['design_ids']),
              'human_rows':prov.get('explicit_human',0),
              'note':'Engine demonstrations only (deterministic_synthetic). No human-reviewed designer pairs or reviewed trajectories exist yet; quality claims require them.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def fidelity_report(demo_paths):
    """design_to_plan -> apply_plan reconstruction error of every engine final, v1 vs v2 (guide 15.3)."""
    rows=[];summary=Counter()
    for p in demo_paths:
        for l in Path(p).read_text().splitlines():
            if not l.strip():continue
            d=json.loads(l);s,t=d['source'],d['target'];A=analyze(s);f=d['final']['design'];rec={'demo_id':d['demo_id'],'design_id':d['design_id'],'target':t,
                'blocks':{type(b).__name__:sum(type(x).__name__==type(b).__name__ for x in A.all_blocks()) for b in A.all_blocks()}}
            for schema,plan in ((PLAN_V1,design_to_plan(f,A)),(PLAN_V2,design_to_plan_v2(f,s,A))):
                o=materialize_plan(s,t,A,plan)
                fid=plan_fidelity(f,o.design) if o.design else {'class':'materialization_'+o.status}
                rec[schema]={**fid,'status':o.status,'reasons':[r.get('name',r['code']) for r in o.reasons][:3]}
                summary[schema+':'+fid['class']]+=1
            rows.append(rec)
    return {'summary':dict(sorted(summary.items())),'cases':rows}


# ----------------------------------------------------------------- validate

def load_rows(data,kind,split='train'):
    path=Path(data)/FILES[(kind,split)]
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def validate_row(row,data=None,excl=None,split_manifest=None):
    """Every reason this row cannot be trained on. Re-materializes through the runtime path."""
    reasons=check_provenance(row)
    for k in ('row_id','kind','source_family','split','source_hash','target','completion','task','prompt'):
        if k not in row:reasons.append('missing_'+k)
    if reasons:return reasons
    task=json.loads(row['task']);source,target=task['source'],task['target']
    if design_hash(source)!=row['source_hash']:reasons.append('source_hash_mismatch')
    if task['target']!=row['target']:reasons.append('target_mismatch')
    if row['split'] not in EXPORT_SPLITS:reasons.append('non_exported_split')
    if split_manifest is not None:
        try:
            if split_of(row['source_family'],split_manifest)!=row['split']:reasons.append('split_mismatch')
        except ValueError:reasons.append('family_not_in_manifest')
    excl=excl or suite_exclusions()
    if excluded(excl,source=source,family=row['source_family'],design_id=row.get('design_id')):reasons.append('visual_suite_source')
    if data is not None:
        for im in row.get('images',[]):
            if not (Path(data)/im).is_file():reasons.append('missing_image')
    A=analyze(source,group_policies=(task.get('options') or row.get('analysis_options') or {}).get('group_policies'))
    if row['kind']=='plan':
        plan=parse_plan_text(row['completion'])
        if plan is None:return reasons+['unparseable_completion']
        o=materialize_plan(source,target,A,plan,row.get('locks') or None)
        if o.status!='ok':return reasons+['plan_'+o.status]
        if not o.raw_valid:reasons.append('raw_invalid_plan')
        if design_hash(o.design)!=row.get('target_revision'):reasons.append('target_revision_mismatch')
        if (row.get('fidelity') or {}).get('class') not in ('exact','near'):reasons.append('lossy_target')
    elif row['kind']=='edit':
        from engine.materialize import apply_edit
        state=task.get('state')
        if state is None:return reasons+['missing_state']
        revision=DesignRevision.create(state)
        if revision.revision_hash!=row.get('base_revision'):reasons.append('base_revision_mismatch')
        if row.get('orientation')!='base_to_result':reasons.append('bad_orientation')
        action=parse_plan_text(row['completion'])
        try:
            result=apply_edit(source,revision,action_to_proposal(action,revision,provenance=row['provenance'],locks=row.get('locks') or None),row.get('locks') or None)
        except (PlanRejected,ValueError,TypeError):return reasons+['edit_not_materializable']
        if design_hash(result)!=row.get('result_revision'):reasons.append('result_revision_mismatch')
    else:reasons.append('unknown_kind')
    return reasons


def validate_dataset(data):
    data=Path(data);excl=suite_exclusions();report={'rows':0,'invalid':0,'reasons':Counter(),'provenance':Counter(),'kinds':Counter()}
    sm=data/'split_manifest.json';split_manifest=json.loads(sm.read_text()) if sm.exists() else None
    for (kind,split) in FILES:
        for row in load_rows(data,kind,split):
            report['rows']+=1;report['provenance'][row.get('provenance')]+=1;report['kinds'][f'{split}/{kind}']+=1
            rs=validate_row(row,data,excl,split_manifest)
            if rs:report['invalid']+=1;report['reasons'].update(rs)
    manifest_path=data/'manifest.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text())
        for rel,digest in {**manifest.get('files',{}),**{'images/'+name:value for name,value in manifest.get('images',{}).items()}}.items():
            path=data/rel
            if Path(rel).is_absolute() or '..' in Path(rel).parts or not path.is_file() or sha256_text(path.read_bytes())!=digest:
                report['invalid']+=1;report['reasons']['dataset_hash_mismatch']+=1
        if sm.is_file() and manifest.get('split_manifest_sha256')!=sha256_text(_dumps(json.loads(sm.read_text()))):
            report['invalid']+=1;report['reasons']['split_manifest_hash_mismatch']+=1
    report={k:(dict(v) if isinstance(v,Counter) else v) for k,v in report.items()}
    report['ok']=report['rows']>0 and report['invalid']==0
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('collect');c.add_argument('corpus',type=Path);c.add_argument('--out',type=Path,required=True);c.add_argument('--limit',type=int)
    c.add_argument('--targets',default='1080x1080,1584x396');c.add_argument('--refine-seconds',type=float,default=4.0);c.add_argument('--min-overall',type=float,default=80.0)
    b=sub.add_parser('build');b.add_argument('demos',type=Path,nargs='+');b.add_argument('--out',type=Path,required=True);b.add_argument('--split-manifest',type=Path)
    b.add_argument('--plan-schema',default=PLAN_V2,choices=[PLAN_V1,PLAN_V2]);b.add_argument('--no-render',action='store_true')
    v=sub.add_parser('validate');v.add_argument('data',type=Path)
    f=sub.add_parser('fidelity');f.add_argument('demos',type=Path,nargs='+');f.add_argument('--out',type=Path)
    a=ap.parse_args()
    if a.cmd=='collect':
        targets=[tuple(map(int,t.split('x'))) for t in a.targets.split(',')]
        print(json.dumps(collect(a.corpus,a.out,targets,a.limit,a.refine_seconds,a.min_overall),indent=2))
    elif a.cmd=='build':
        sm=json.loads(a.split_manifest.read_text()) if a.split_manifest else None
        print(json.dumps(build(a.demos,a.out,sm,a.plan_schema,not a.no_render),indent=2))
    elif a.cmd=='fidelity':
        r=fidelity_report(a.demos)
        if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
        print(json.dumps(r['summary'],indent=2))
    else:
        r=validate_dataset(a.data);print(json.dumps(r,indent=2));raise SystemExit(0 if r['ok'] else 1)
