"""Build VLM training sets whose targets all pass the guardrails.

  python -m training.export CORPUS_DIR --out output/training/vlm [--families families.jsonl]

Writes, per split (train / validation; test families are never exported):
  sft.jsonl   prompt + source image -> plan JSON   (designer resizes, then the best engine plan that
              still passes every rubric gate after apply_plan)
  dpo.jsonl   prompt + chosen/rejected plans       (both pass Tier 0-1; reward gap >= margin)
  grpo.jsonl  prompt + task for training.reward.grpo_reward
The prompt is engine.neural_proposals.plan_prompt, identical to runtime.
Public challenge samples are excluded unless --allow-samples.
"""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from engine.analysis import analyze
from engine.canonical import ASSETS_DIR, ROOT
from engine.formats import CATALOGUE, aspect_bucket
from engine.neural_proposals import PLAN_V2, arrangement_of, design_to_plan_v2, materialize_plan, plan_fidelity, plan_prompt_v2
from engine.contracts import design_hash
from training.reward import assess, reward_from, version_identity


from training.families import split_of, resolve_family, build_split_manifest, immutable_signature, refuse_qa_corpus

def _hash(design):
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()


def public_samples():
    return {_hash(json.loads(p.read_text())['source']) for p in (ROOT / 'input').glob('*.json')}


def _image(source,folder):
    from engine.quality import resource_fingerprint,verify_design
    name=hashlib.sha256((_hash(source)+resource_fingerprint(source)).encode()).hexdigest()[:24]+'.png';path=folder/name
    if not path.exists():
        verification=verify_design(source,output_path=path,max_size=672)
        if not verification['verified']:raise ValueError('Training source render unavailable: '+verification.get('error','unknown'))
    return 'images/'+name


def _row(source, target, A, image, family, options=None, locks=None):
    return {'family': str(family),'source_family':str(family),'plan_schema':PLAN_V2,'locks':locks or {},'analysis_options':{'group_policies':(options or {}).get('group_policies')},'source_hash':design_hash(source),'versions':version_identity(), 'bucket': aspect_bucket(source['canvas'], target), 'images': [image],
            'prompt': plan_prompt_v2(source, target, A, locks=locks), 'task': json.dumps({'source': source, 'target': target,'options':options or {}})}


def export(corpus, out, families=None, formats=None, min_overall=85.0, margin=0.05, allow_samples=False, limit=None, max_candidates=8, split_manifest=None):
    from engine.pipeline import generate
    refuse_qa_corpus(corpus)
    out.mkdir(parents=True, exist_ok=True); images = out / 'images'; images.mkdir(exist_ok=True)
    banned = set() if allow_samples else public_samples(); files = {}; counts = Counter()
    # The repeatedly reviewed visual panel is never training data, even when public samples are allowed.
    panel_signatures = set()
    panel = ROOT / 'docs/individual-update-plans/visual-suite.v1.json'
    if panel.exists():
        for spec in json.loads(panel.read_text())['sources']:
            path = ROOT / spec['path']
            if path.exists():
                item = json.loads(path.read_text()); panel_source = item.get('source', item); banned.add(_hash(panel_source)); panel_signatures.add(immutable_signature(panel_source))
    corpus_records = []
    for path in sorted(Path(corpus).glob('*.json')):
        try:
            item = json.loads(path.read_text())
            if item.get('source', item).get('elements'): corpus_records.append((path.stem, item))
        except (ValueError, TypeError, AttributeError): pass
    family_rows = [json.loads(line) for line in Path(families).read_text().splitlines() if line.strip()] if families else []
    manifest = json.loads(Path(split_manifest).read_text()) if split_manifest else build_split_manifest(corpus_records + [(p['family'], p) for p in family_rows])
    (out / 'family-splits.json').write_text(json.dumps(manifest, indent=2))
    def write(split, kind, row):
        key = (split, kind)
        if key not in files:
            files[key]=[]
        files[key].append(row);counts[f'{split}/{kind}']+=1
        if row.get('provenance')=='explicit_human':counts['explicit_human_exported']+=1
    targets = formats or [(w, h) for _, w, h in CATALOGUE]
    for p in family_rows:
        split = split_of(p['family'], manifest)
        if split in ('test', 'calibration') or _hash(p['source']) in banned or immutable_signature(p['source']) in panel_signatures:
            continue
        review=p.get('review') or {}
        if not review.get('reviewer_id') or not review.get('committed_at') or review.get('verdict') not in ('accepted','approved','A','B'):
            counts['quarantined_unreviewed_designer']+=1;continue
        options=p.get('options') or p.get('teacher_options') or {};locks=p.get('locks') or {};A=analyze(p['source'],group_policies=options.get('group_policies'));plan=design_to_plan_v2(p['reference'],p['source'],A)
        outcome=materialize_plan(p['source'],p['target_canvas'],A,plan,locks)
        if outcome.status!='ok' or not outcome.raw_valid or plan_fidelity(p['reference'],outcome.design)['class']=='lossy':
            counts['quarantined_lossy_designer']+=1;continue
        row=_row(p['source'],p['target_canvas'],A,_image(p['source'],images),p['family'],options,locks)
        write(split,'sft',{**row,'completion':json.dumps(plan,sort_keys=True),'origin':'designer','provenance':'explicit_human',
                           'review':review,'raw_valid':True,'repaired_valid':outcome.repaired_valid,'fidelity':plan_fidelity(p['reference'],outcome.design)})
    sources = sorted(Path(corpus).glob('*.json'))[:limit]
    for path in sources:
        try:
            task = json.loads(path.read_text()); source = task.get('source', task)
        except ValueError:
            continue
        family = resolve_family(task, path.stem); split = split_of(family, manifest)
        if split in ('test','calibration') or _hash(source) in banned or immutable_signature(source) in panel_signatures:
            counts['skipped_' + (split if split in ('test','calibration') else 'public_sample_or_reviewed_panel')] += 1; continue
        options=task.get('options') or {};locks=task.get('locks') or {};A = analyze(source,group_policies=options.get('group_policies')); image = _image(source, images)
        for w, h in targets:
            target = {'width': w, 'height': h}
            if source['canvas'] == target:
                continue
            cands = [c for c in generate(source, target, options={**options,'render':False}, locks=locks, all_candidates=True) if c['valid'] and c['rubric']['gate']]
            cands.sort(key=lambda c: -c['rubric']['tier2']['overall']); scored = {}
            # Judge each plan by what it becomes through apply_plan: that is what the model will output.
            for c in cands[:max_candidates]:
                plan=json.dumps(design_to_plan_v2(c['design'],source,A,arrangement_of(c['template'])),sort_keys=True)
                if plan not in scored:
                    outcome=materialize_plan(source,target,A,json.loads(plan),locks)
                    if outcome.status!='ok' or not outcome.raw_valid:
                        counts['raw_invalid_candidate']+=1;continue
                    if plan_fidelity(c['design'],outcome.design)['class']=='lossy':
                        counts['lossy_candidate']+=1;continue
                    result=assess(source,outcome.design,A,target)
                    scored[plan]=(reward_from(result),result['rubric'],c['template'])
            ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])
            if not ranked or ranked[0][1][0] <= 0:
                counts['no_gated_candidate'] += 1; continue
            (best, (r_best, rub_best, template)), row = ranked[0], _row(source, target, A, image, family,options,locks)
            clean = rub_best['passed'] and rub_best['tier2']['overall'] >= min_overall
            if clean:
                write(split, 'sft', {**row, 'completion': best, 'origin':'engine','provenance':'deterministic_synthetic','teacher':'deterministic engine','raw_valid':True,'repaired_valid':True,'template':template,'reward':r_best})
            for plan, (r, rub, _) in [x for x in ranked[1:] if r_best - x[1][0] >= margin][:2] if clean else []:
                write(split, 'dpo', {**row, 'chosen': best, 'rejected': plan, 'reward_chosen':r_best,'reward_rejected':r,'provenance':'deterministic_synthetic','teacher':'deterministic rubric','raw_valid':True,'repaired_valid':True})
            write(split,'grpo',{**row,'provenance':'deterministic_synthetic','teacher':'deterministic engine task corpus'})
        print(path.stem, dict(counts), flush=True)
    for split in ('train','validation'):
        for kind in ('sft','dpo','grpo'):
            path=out/f'{kind}{"" if split=="train" else "."+split}.jsonl'
            temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(row)+'\n' for row in files.get((split,kind),[])));temp.replace(path)
    manifest = {'dataset_schema':'legacy-vlm-export/2','plan_schema':PLAN_V2,'versions':version_identity(),'human_rows':counts.get('explicit_human_exported',0),'counts': dict(counts), 'min_overall': min_overall, 'dpo_reward_margin': margin, 'formats': targets, 'corpus': str(corpus),
                'families_file': str(families) if families else None, 'excluded_public_samples': not allow_samples,
                'split_policy': 'frozen family-splits.json; test/calibration and reviewed visual panel never exported', 'split_manifest_sha256': hashlib.sha256((out / 'family-splits.json').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('corpus', type=Path); ap.add_argument('--out', type=Path, default=Path('output/training/vlm'))
    ap.add_argument('--split-manifest', type=Path); ap.add_argument('--families', type=Path); ap.add_argument('--formats'); ap.add_argument('--min-overall', type=float, default=85.0)
    ap.add_argument('--margin', type=float, default=0.05, help='minimum reward gap for DPO pairs'); ap.add_argument('--allow-samples', action='store_true'); ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    formats = [tuple(map(int, f.split('x'))) for f in a.formats.split(',')] if a.formats else None
    print(json.dumps(export(a.corpus, a.out, a.families, formats, a.min_overall, a.margin, a.allow_samples, a.limit, split_manifest=a.split_manifest), indent=2))
