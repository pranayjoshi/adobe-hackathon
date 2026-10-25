"""Export durable, explicit human pair/correction feedback for train_ranker.

Unary likes/dislikes and behavioral acceptance remain unary. No model training
or artifact promotion occurs here. Exported test/calibration partitions are
for held-out evaluation and must be used with the emitted split manifest.
"""
from __future__ import annotations
import argparse, json, hashlib
from collections import Counter
from pathlib import Path
from engine.canonical import ROOT
from engine.contracts import design_hash
from service.storage import Repository
from training.families import build_split_manifest, immutable_signature, resolve_family, split_of
from evaluation.visual_suite import atomic_json, digest


def reviewed_panel_signatures():
    suite=ROOT/'docs/individual-update-plans/visual-suite.v1.json'
    values=set()
    if suite.is_file():
        for item in json.loads(suite.read_text())['sources']:
            path=ROOT/item['path']
            if path.is_file():
                record=json.loads(path.read_text());values.add(immutable_signature(record.get('source',record)))
    return values


def _preview_path(repo,rid,index,revision):
    hashed=revision['revision_hash'];local=repo.root/rid/str(index)/(hashed+'.png')
    if local.is_file():return local
    url=revision.get('render','')
    if url.startswith('/renders/'):
        name=url.removeprefix('/renders/');path=repo.root.parent/'renders'/name
        if Path(name).name==name and path.is_file():return path
    return None


def _preview_exists(repo,rid,index,revision):
    path=_preview_path(repo,rid,index,revision)
    if path is None:return False
    from PIL import Image
    try:
        with Image.open(path) as im:
            expected=revision['design']['canvas']
            if im.size!=(expected['width'],expected['height']):return False
            im.verify()
        return True
    except (OSError,ValueError):return False


def export_preferences(workspace,out,split_manifest=None):
    if (Path(workspace)/'QA_ONLY').exists(): raise ValueError('QA_ONLY workspace contains agent test feedback, never human training labels')
    repo=Repository(workspace);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with repo.connect() as db:
        feedback=[json.loads(row[0]) for row in db.execute('SELECT body FROM feedback ORDER BY rowid')]
    counts=Counter({'seen':0,'exported':0});eligible=[];runs={};excluded=reviewed_panel_signatures()
    for vote in feedback:
        counts['seen']+=1
        if vote.get('provenance')!='human':counts['non_human_or_behavioral']+=1;continue
        if vote['verdict'] not in ('a','b','correction'):counts['unary_or_nondecisive']+=1;continue
        rid=vote['run_id'];run=repo.run(rid)
        if not run:counts['missing_run']+=1;continue
        source=run['request']['source']
        if immutable_signature(source) in excluded:counts['reviewed_visual_panel']+=1;continue
        context=vote.get('context') or {};target=context.get('target');hashed=vote['revision_hash'];other=vote.get('other_hash')
        if not other or hashed==other or context.get('source_hash')!=design_hash(source):counts['invalid_context_or_pair']+=1;continue
        index=vote['target_index'];a=repo.revision(rid,index,hashed);b=repo.revision(rid,index,other)
        if not a or not b:counts['missing_revision']+=1;continue
        if any(not side.get('verified') or not side.get('render') or not _preview_exists(repo,rid,index,side) for side in (a,b)):
            counts['unverified_or_missing_preview']+=1;continue
        if any(side['design']['canvas']!=target or design_hash(side['design'])!=side['revision_hash'] for side in (a,b)):
            counts['revision_identity_or_target_mismatch']+=1;continue
        if vote['verdict']=='correction' and (a.get('operation')!='human_correction' or a.get('parent_revision')!=other or not a.get('accepted')):
            counts['uncommitted_correction']+=1;continue
        family=resolve_family({'family':context.get('family') or run.get('family')})
        runs[rid]={'family':family,'source':source};eligible.append((vote,run,a,b,family))
    manifest=json.loads(Path(split_manifest).read_text()) if split_manifest else build_split_manifest(list(runs.items()))
    manifest_path=out.with_suffix('.splits.json');atomic_json(manifest_path,manifest)
    split_hash=digest(manifest_path)
    # Raw callbacks can lack rank features. Reconstruct deterministically without
    # rendering; only the actual human-observed verified revisions are eligible.
    from engine.pipeline import assess_design
    from engine.analysis import analyze
    evaluator_paths=[ROOT/'engine'/name for name in ('pipeline.py','metrics.py','quality.py','rubric.py','judging.py','typography.py',
        'analysis.py','blocks.py','relations.py','group_policies.py','scene.py','photo_analysis.py','learning.py')]
    evaluator_version=hashlib.sha256(json.dumps([(p.name,digest(p)) for p in evaluator_paths],sort_keys=True).encode()).hexdigest()
    rows=[]
    for vote,run,a,b,family in eligible:
        try:split=split_of(family,manifest)
        except ValueError:counts['unknown_or_excluded_family']+=1;continue
        source=run['request']['source'];options=run['request'].get('options',{})
        analysis=analyze(source,group_policies=options.get('group_policies'),
                         overrides=run['request'].get('analysis_overrides') or None);sides=[]
        from engine.learning import apply_roles
        apply_roles(source,analysis,enabled=options.get('role_mode',options.get('roles')))
        from engine.relations import GRAPH_VERSION
        from engine.group_policies import POLICY_VERSION
        for revision in (a,b):
            candidate=assess_design(source,revision['design']['canvas'],revision['design'],analysis,'Human preference export',run['request'].get('locks'))
            candidate['revision_hash']=revision['revision_hash'];sides.append(candidate)
        if any(not side['valid'] or not side['rubric']['gate'] for side in sides):counts['structural_gate_failed']+=1;continue
        row={'pair':'sqlite:'+vote['id'],'kind':'human','provenance':'human','family':family,'split':split,
             'decision':'a' if vote['verdict']=='correction' else vote['verdict'],'a':sides[0],'b':sides[1],
             'a_hash':a['revision_hash'],'b_hash':b['revision_hash'],'a_preview_sha256':digest(_preview_path(repo,vote['run_id'],vote['target_index'],a)),'b_preview_sha256':digest(_preview_path(repo,vote['run_id'],vote['target_index'],b)),'source_hash':design_hash(source),'target':a['design']['canvas'],
             'reviewer_id':vote.get('reviewer_id'),'context':vote.get('context',{}),'reason':vote.get('reason'),'created':vote['created'],
             'feedback_id':vote['id'],'feedback_type':vote['verdict'],'run_id':vote['run_id'],'target_index':vote['target_index'],
             'feature_provenance':'deterministic_reassessment_of_verified_human_observed_revisions','evaluator_version':evaluator_version,
             'analysis_graph_version':GRAPH_VERSION,'group_policy_version':POLICY_VERSION,'generation_options':options,
             'analysis_overrides':run['request'].get('analysis_overrides',{}),
             'split_manifest_sha256':split_hash}
        rows.append(row);counts['exported']+=1
    temp=out.with_suffix(out.suffix+'.tmp');temp.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows));temp.replace(out)
    report={'schema_version':1,'counts':dict(counts),'output':str(out),'output_sha256':digest(out),'split_manifest':str(manifest_path),
            'split_manifest_sha256':split_hash,'evaluator_version':evaluator_version,'exporter_sha256':digest(__file__),
            'training_performed':False,'human_labels_sufficient_for_training':len(rows)>=20 and len({r['family'] for r in rows})>=5,
            'note':'Availability threshold only; independent held-out human evaluation and explicit promotion remain required.'}
    atomic_json(out.with_suffix('.manifest.json'),report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--workspace',type=Path,default=ROOT/'output/workspace');p.add_argument('--out',type=Path,default=ROOT/'output/training/human-preferences.jsonl');p.add_argument('--split-manifest',type=Path)
    a=p.parse_args();print(json.dumps(export_preferences(a.workspace,a.out,a.split_manifest),indent=2))
