"""LoRA fine-tuning of the layout proposer: SFT, then DPO or GRPO.

SFT (guide 15.6) now runs through training.vlm_adapters.train_sft: explicit assistant-only
label masking (asserted per row), LoRA on language-model projections, MPS/CPU without
CUDA/bitsandbytes assumptions, and an engine.vlm-checkable export bundle. It accepts both
training.edit_dataset directories (manifest dataset_schema chameleon-learn-dataset/1) and
legacy training.export directories (rows mapped to provenance; unreviewed designer rows refused).

  python -m training.train_vlm sft output/learn/data --dry-run [--tokenize --adapter tiny-qwen2.5-vl-test]
  python -m training.train_vlm sft output/learn/data --adapter qwen2.5-vl-3b-instruct --out work/vlm-sft --max-steps 20 --rank 16 --lr 1e-4

  python -m training.train_vlm sft  output/training/vlm --dry-run
  python -m training.train_vlm sft  output/training/vlm --base Qwen/Qwen2.5-VL-3B-Instruct --out work/vlm-sft
  python -m training.train_vlm dpo  output/learn-dpo/data/human_only --dry-run         # training.edit_dpo dataset (re-materialized, audited)
  python -m training.train_vlm dpo  output/learn-dpo/data/human_only --sft-bundle work/vlm-sft/bundle --out work/vlm-dpo --lr 1e-5 --beta 0.1
  python -m training.train_vlm grpo output/training/vlm --base work/vlm-sft/merged --out work/vlm-grpo

Data comes from `python -m training.export`. --dry-run checks every row
(images exist, plans parse, every block appears once, rewards ordered)
without ML dependencies. Training needs requirements-train.txt and a GPU or
Apple-silicon machine. GRPO rewards are training.reward.grpo_reward, i.e. the
evaluation rubric itself.

Adapters must be published (public Hugging Face repo) before a submission
relies on them (RULES.md). The runtime engine uses llama.cpp: merge the
adapter (--merge) and convert to GGUF plus projector before setting
CHAMELEON_VLM_MODEL / CHAMELEON_VLM_MMPROJ.
"""
import argparse, json, statistics
from pathlib import Path
from engine.analysis import analyze
from engine.constitution import PLAN_SCHEMA
from engine.neural_proposals import plan_valid
from training.reward import parse_plan, reward

FILES = {'sft': 'sft', 'dpo': 'dpo', 'grpo': 'grpo'}


def rows(data, stage, split='train'):
    path = Path(data) / (FILES[stage] + ('' if split == 'train' else '.' + split) + '.jsonl')
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def check(data, stage, rewards=True, limit=None):
    """Dry run: validate a dataset against the same contracts the engine enforces.

    DPO semantics (guide 16.3): both sides are always structurally validated. A synthetic/rubric pair
    (legacy training.export rows carry no provenance and are rubric-derived) whose chosen side does not
    out-score the rejected side is misordered. A decisive human (explicit_human/behavioral) pair is NOT
    rejected for that: it is flagged and counted in ``reward_disagreements`` (the surrogate is not ground
    truth for designer preference). Human rows still need a review record; tie/both_bad/unsure rows must
    never appear as binary rows.
    """
    if stage == 'dpo':
        from training.edit_dpo import is_dpo_dataset, validate_dataset
        if is_dpo_dataset(data):
            r = validate_dataset(data, audit=rewards); r['stage'] = 'dpo'; return r
    import jsonschema
    from training.edit_dataset import check_provenance
    from training.edit_dpo import DECISIVE, audit_pair
    report = {'stage': stage, 'rows': 0, 'missing_images': 0, 'invalid_plans': 0, 'rewards': [], 'misordered_pairs': 0}
    if stage == 'dpo':
        report.update(reward_disagreements=0, disagreement_rows=[], bad_provenance=0, non_binary_rows=0)
    for split in ('train', 'validation'):
        for r in rows(data, stage, split)[:limit]:
            report['rows'] += 1; task = json.loads(r['task']); A = analyze(task['source'])
            if not all((Path(data) / im).exists() for im in r['images']):
                report['missing_images'] += 1
            provenance = r.get('provenance', 'deterministic_synthetic')
            if stage == 'dpo':
                if 'provenance' in r and check_provenance(r):
                    report['bad_provenance'] += 1
                if r.get('verdict', 'A') not in DECISIVE:
                    report['non_binary_rows'] += 1
            plans = [r[k] for k in ('completion', 'chosen', 'rejected') if k in r]
            for text in plans:
                plan = parse_plan(text)
                try:
                    from engine.neural_proposals import validate_plan,materialize_plan
                    check=validate_plan(A,plan,source=task['source'],target=task['target'],locks=r.get('locks'))
                    outcome=materialize_plan(task['source'],task['target'],A,plan,r.get('locks')) if check.ok else None
                    ok=check.ok and outcome.status=='ok' and outcome.raw_valid
                except (jsonschema.ValidationError, TypeError, KeyError):
                    ok = False
                report['invalid_plans'] += not ok
            if rewards and plans:
                scores = [reward(task['source'], task['target'], parse_plan(t), A) for t in plans]
                human = stage == 'dpo' and provenance not in ('teacher_model', 'deterministic_synthetic')
                if not human:
                    report['rewards'].append(scores[0])
                if stage == 'dpo':
                    verdict = audit_pair(provenance, scores[0], scores[1])
                    if verdict == 'misordered':
                        report['misordered_pairs'] += 1
                    elif verdict == 'reward_disagreement':
                        report['reward_disagreements'] += 1
                        report['disagreement_rows'].append({'row': r.get('row_id') or r.get('pref_id'), 'provenance': provenance, 'chosen': scores[0], 'rejected': scores[1]})
    rs = report.pop('rewards')
    report['reward_median'] = statistics.median(rs) if rs else None
    report['reward_min'] = min(rs) if rs else None
    clean = not report.get('bad_provenance') and not report.get('non_binary_rows')
    report['ok'] = clean and report['rows'] > 0 and not report['missing_images'] and not report['invalid_plans'] and not report['misordered_pairs'] and (stage == 'grpo' or (stage == 'dpo' and rewards and not rs) or (report['reward_min'] or 0) > 0)
    return report


LEARN_SCHEMA = 'chameleon-learn-dataset/1'


def is_learn_dataset(data):
    m = Path(data) / 'manifest.json'
    return m.exists() and json.loads(m.read_text()).get('dataset_schema') == LEARN_SCHEMA


def legacy_sft_rows(data, split):
    """training.export sft rows -> learn-format plan rows with explicit provenance."""
    import hashlib
    out = []
    for r in rows(data, 'sft', split):
        origin = r.get('origin')
        if origin == 'designer' and r.get('provenance')!='explicit_human':
            continue  # mined designer resizes are human-made but unreviewed: never auto-labelled explicit_human
        out.append({**r, 'kind':'plan','provenance':r.get('provenance','deterministic_synthetic'),'teacher':r.get('teacher','training.export ('+str(origin)+')'),
                    'row_id': hashlib.sha256((r['prompt'] + r['completion']).encode()).hexdigest()[:20], 'plan_schema':r.get('plan_schema','layout-plan/1')})
    return out


def sft_rows(data, split, kinds=('plan', 'edit')):
    if is_learn_dataset(data):
        from training.edit_dataset import load_rows
        return [r for k in kinds for r in load_rows(data, k, split)]
    return legacy_sft_rows(data, split)


def check_learn(data, adapter_id=None, tokenize=False, limit=None):
    """Dry run for learn datasets: provenance/split/suite/rematerialization + adapter + optional masking."""
    from training.edit_dataset import validate_dataset
    from engine.vlm import ADAPTERS
    report = validate_dataset(data)
    if adapter_id:
        a = ADAPTERS.get(adapter_id)
        manifest = json.loads((Path(data) / 'manifest.json').read_text())
        report['adapter'] = {'id': adapter_id, 'known': a is not None,
                             'schema_ok': bool(a) and manifest.get('plan_schema') in a.plan_schemas and manifest.get('action_schema') in a.plan_schemas,
                             'transform_ok': bool(a) and manifest.get('image_transform') == a.image_transform}
        report['ok'] = report['ok'] and report['adapter']['known'] and report['adapter']['schema_ok'] and report['adapter']['transform_ok']
        if tokenize and a:
            from training.vlm_adapters import load_base, encode_example, assert_supervision, _load_images
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(a.hf_repo, revision=a.revision, local_files_only=True)
            except Exception as exc:
                report['tokenize'] = {'status': 'pending', 'prerequisite': 'processor for %s not cached locally (%s)' % (a.hf_repo, type(exc).__name__)}
                return report
            n = 0; lengths = []
            for r in sft_rows(data, 'train')[:limit]:
                ex = encode_example(processor, _load_images(r, data), r['prompt'], r['completion']); assert_supervision(processor, ex)
                n += 1; lengths.append(int(ex['input_ids'].shape[0]))
            report['tokenize'] = {'status': 'ok', 'rows': n, 'max_tokens': max(lengths) if lengths else None, 'median_tokens': statistics.median(lengths) if lengths else None}
    return report


def _messages(r, data):
    from PIL import Image
    user = {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': r['prompt']}]}
    say = lambda text: [{'role': 'assistant', 'content': [{'type': 'text', 'text': text}]}]
    from engine.vlm import prepare_image
    with Image.open(Path(data) / r['images'][0]) as im:
        image = prepare_image(im)
    return user, say, image


def dataset(data, stage, split):
    from datasets import Dataset
    out = []
    for r in rows(data, stage, split):
        user, say, image = _messages(r, data)
        if stage == 'sft':
            out.append({'messages': [user] + say(r['completion']), 'images': [image]})
        elif stage == 'dpo':
            out.append({'prompt': [user], 'chosen': say(r['chosen']), 'rejected': say(r['rejected']), 'images': [image]})
        else:
            out.append({'prompt': [user], 'images': [image], 'task': r['task']})
    return Dataset.from_list(out) if out else None


def train(stage, data, base, out, epochs=2, lr=1e-4, rank=16, batch=1, accum=8, merge=False, max_steps=-1, max_completion=1024, cpu=False, adapter_id=None, device='auto', sft_bundle=None,beta=.1,grpo_evidence=None):
    if stage == 'sft':
        from training.vlm_adapters import train_sft
        from engine.vlm import ADAPTERS
        aid = adapter_id or next((k for k, v in ADAPTERS.items() if v.hf_repo == base), None)
        if aid is None:
            raise SystemExit('unregistered base model %r: add an engine.vlm.ModelAdapter (or pass --adapter with --base as a local path)' % base)
        manifest = json.loads((Path(data) / 'manifest.json').read_text()) if (Path(data) / 'manifest.json').exists() else None
        local = base if base and Path(base).exists() else None
        return train_sft(sft_rows(data, 'train'), sft_rows(data, 'validation'), data, aid, out, base=local, rank=rank, lr=lr,
                         max_steps=max_steps if max_steps > 0 else 10 ** 9, epochs=epochs, accum=accum, device='cpu' if cpu else device, dataset_manifest=manifest)
    if stage == 'dpo':
        # The old path passed the *base* model + a fresh peft_config to TRL's DPOTrainer, so its implicit reference
        # was the disabled-adapter base model, not the SFT policy. DPO now always starts from an exported SFT bundle
        # with a frozen copy of that adapter as reference (training.edit_dpo.train_dpo).
        from training.edit_dpo import is_dpo_dataset, load_pairs, train_dpo
        if not is_dpo_dataset(data):
            raise SystemExit('dpo needs a training.edit_dpo dataset: run `python -m training.edit_dpo import-legacy %s --out prefs.jsonl` then `build`' % data)
        if not sft_bundle:
            raise SystemExit('dpo needs --sft-bundle <exported SFT bundle dir> (the frozen reference and the policy init)')
        man = json.loads((Path(data) / 'manifest.json').read_text())
        tr = load_pairs(data, 'plan', 'train') + load_pairs(data, 'edit', 'train'); va = load_pairs(data, 'plan', 'validation') + load_pairs(data, 'edit', 'validation')
        return train_dpo(tr, va, data, sft_bundle, out, beta=beta, lr=lr, max_steps=max_steps if max_steps > 0 else 10 ** 9, epochs=epochs, accum=accum,
                         device='cpu' if cpu else device, dataset_manifest=man)
    if stage=='grpo':
        from training.grpo_training import train as train_grpo,GRPOConfig
        return train_grpo(data,sft_bundle,out,evidence=grpo_evidence,
             config=GRPOConfig(beta=beta,lr=lr,max_steps=max_steps if max_steps>0 else 20,max_new_tokens=max_completion),device='cpu' if cpu or device=='auto' else device)
    raise ValueError('Unknown training stage')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=list(FILES)); ap.add_argument('data', type=Path)
    ap.add_argument('--base', default='Qwen/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--out', type=Path, default=Path('work/vlm'))
    ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--lr',type=float); ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--adapter', help='engine.vlm.ADAPTERS id'); ap.add_argument('--device', default='auto'); ap.add_argument('--tokenize', action='store_true', help='dry run: also assert loss masking with the cached processor')
    ap.add_argument('--merge', action='store_true'); ap.add_argument('--max-steps', type=int, default=-1, help='smoke runs'); ap.add_argument('--max-completion', type=int, default=1024); ap.add_argument('--cpu', action='store_true', help='GRPO on Apple MPS stalled in TRL 1.16 smoke runs'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--limit', type=int)
    ap.add_argument('--grpo-evidence',type=Path,help='reviewed/calibration/independent-eval/rejection-distillation artifact paths JSON');ap.add_argument('--sft-bundle', type=Path, help='dpo: exported SFT bundle (policy init + frozen reference)'); ap.add_argument('--beta',type=float)
    a=ap.parse_args()
    if a.lr is None:a.lr=1e-5 if a.stage=='dpo' else 5e-6 if a.stage=='grpo' else 1e-4
    if a.beta is None:a.beta=.02 if a.stage=='grpo' else .1
    evidence=json.loads(a.grpo_evidence.read_text()) if a.grpo_evidence else {}
    if a.dry_run and a.stage=='grpo':
        from training.grpo_training import prerequisites
        print(json.dumps(prerequisites(a.data,a.sft_bundle,evidence),indent=2));raise SystemExit(0)
    if a.dry_run and a.stage == 'sft' and is_learn_dataset(a.data):
        r = check_learn(a.data, a.adapter, a.tokenize, a.limit); print(json.dumps(r, indent=2, default=str)); raise SystemExit(0 if r['ok'] else 1)
    if a.dry_run:
        r = check(a.data, a.stage, limit=a.limit); print(json.dumps(r, indent=2)); raise SystemExit(0 if r['ok'] else 1)
    a.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(train(a.stage, a.data, a.base, a.out, a.epochs, a.lr, a.rank, merge=a.merge, max_steps=a.max_steps, max_completion=a.max_completion, cpu=a.cpu, adapter_id=a.adapter, device=a.device, sft_bundle=a.sft_bundle,beta=a.beta,grpo_evidence=evidence), indent=2, default=str))
