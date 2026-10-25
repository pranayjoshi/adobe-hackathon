"""Registered whole-plan GRPO infrastructure with explicit conditional gates.

No command downloads weights. Dry runs and deterministic rollout/kernel smoke
are reproducible infrastructure checks, never evidence of policy improvement.
The production stage requires reviewed data, a reliable frozen SFT policy and
independent evaluation/calibration artifacts; raw rubric reward cannot certify it.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from dataclasses import dataclass,asdict
from pathlib import Path
from training.reward import reward_record,version_identity
from engine.contracts import design_hash

@dataclass(frozen=True)
class GRPOConfig:
    group_size:int=4
    beta:float=.02
    clip_low:float=.2
    clip_high:float=.2
    temperature:float=.8
    top_p:float=.95
    max_new_tokens:int=256
    max_steps:int=20
    lr:float=5e-6
    seed:int=7
    loss_type:str='grpo_token_clip'
    reward_scaling:str='within_prompt_population_std'
    repair:bool=False
    repair_seconds:float=.1
    def __post_init__(self):
        for name in ('beta','clip_low','clip_high','temperature','top_p','lr','repair_seconds'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError('Invalid '+name)
        if not .01<=self.temperature<=2 or not 0<self.top_p<=1:raise ValueError('Invalid sampling configuration')
        if any(type(getattr(self,name)) is not int for name in ('group_size','max_steps','max_new_tokens','seed')):raise ValueError('Integer training budgets/seed required')
        if not 2<=self.group_size<=16 or not 1<=self.max_steps<=10000 or not 1<=self.max_new_tokens<=4096:raise ValueError('Invalid training budget')
        if self.loss_type!='grpo_token_clip' or self.reward_scaling!='within_prompt_population_std':raise ValueError('Unsupported loss/normalization')


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def group_advantages(rewards):
    if len(rewards)<2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in rewards):raise ValueError('Finite reward group required')
    mean=sum(rewards)/len(rewards);variance=sum((x-mean)**2 for x in rewards)/len(rewards)
    scale=math.sqrt(variance)
    return [0. for x in rewards] if scale<=1e-8 else [(x-mean)/scale for x in rewards]


def grpo_loss(logp,old_logp,reference_logp,advantages,mask,config=GRPOConfig()):
    """Token-wise clipped objective, completion-only mask, explicit reverse-KL estimator."""
    import torch
    if logp.shape!=mask.shape or old_logp.shape!=logp.shape or reference_logp.shape!=logp.shape:raise ValueError('Token tensor/mask shape mismatch')
    if advantages.shape!=(logp.shape[0],):raise ValueError('One normalized advantage per completion required')
    if (mask.sum(dim=1)<=0).any():raise ValueError('Empty supervised completion')
    ratio=torch.exp((logp-old_logp.detach()).clamp(-20,20));adv=advantages[:,None]
    clipped=ratio.clamp(1-config.clip_low,1+config.clip_high)
    objective=torch.minimum(ratio*adv,clipped*adv)
    delta=(reference_logp.detach()-logp).clamp(-20,20);kl=torch.exp(delta)-delta-1
    per_completion=((-objective+config.beta*kl)*mask).sum(dim=1)/mask.sum(dim=1)
    loss=per_completion.mean()
    if not torch.isfinite(loss):raise ValueError('Non-finite GRPO loss')
    return loss,{'kl':float(((kl*mask).sum()/mask.sum()).detach()),'clip_fraction':float((((ratio!=clipped)*mask).sum()/mask.sum()).detach())}


def rollout(source,target,completions,*,locks=None,config=GRPOConfig(),cancelled=None,deadline=None,group_policies=None):
    from engine.analysis import analyze
    from training.reward import parse_plan,RewardCache
    cache=RewardCache()
    A=analyze(source,group_policies=group_policies);records=[];started=time.monotonic();status='complete'
    for completion in completions:
        if cancelled and cancelled():status='cancelled';break
        if deadline is not None and time.monotonic()>=deadline:status='deadline';break
        record=cache.evaluate(source,target,parse_plan(completion),A,analysis_options={'group_policies':group_policies},locks=locks,repair=config.repair,repair_seconds=config.repair_seconds,cancelled=cancelled)
        records.append(record)
    rewards=[r['reward'] for r in records]
    advantages=group_advantages(rewards) if len(rewards)>=2 else []
    return {'schema':'grpo-rollout/1','status':status,'source_hash':design_hash(source),'target':target,'reward_identity':version_identity(),
            'config':asdict(config),'records':records,'rewards':rewards,'advantages':advantages,
            'zero_reward_variance':bool(advantages) and not any(advantages),'seconds':round(time.monotonic()-started,6),
            'resource_usage':{'model_calls':0,'renders':sum(r.get('renders',0) for r in records)}}


def _task_rows(data,split):
    from training.edit_dataset import load_rows
    data=Path(data);manifest=json.loads((data/'manifest.json').read_text())
    if manifest.get('dataset_schema')!='legacy-vlm-export/2':return load_rows(data,'plan',split)
    path=data/('grpo'+('' if split=='train' else '.'+split)+'.jsonl')
    return [{**row,'kind':'plan','row_id':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]} for row in (json.loads(x) for x in path.read_text().splitlines() if x.strip())] if path.is_file() else []


def _validate_legacy_tasks(data):
    from training.edit_dataset import suite_exclusions,excluded
    from training.families import split_of
    data=Path(data);splits=json.loads((data/'family-splits.json').read_text());excl=suite_exclusions();errors=[];count=0
    for split in ('train','validation'):
        for row in _task_rows(data,split):
            count+=1;task=json.loads(row['task']);source=task['source'];family=row.get('source_family') or row['family']
            if row.get('plan_schema')!='layout-plan/2' or split_of(family,splits)!=split:errors.append('schema_or_split')
            if excluded(excl,source=source,family=family):errors.append('reviewed_panel')
            if not row.get('images') or any(not (data/path).is_file() for path in row['images']):errors.append('missing_image')
    return {'ok':bool(count) and not errors,'rows':count,'errors':errors}


def prerequisites(data,sft_bundle=None,evidence=None,config=GRPOConfig()):
    """Fail closed on missing evidence. This does not infer human review from quality scores."""
    from engine.vlm import check_bundle,ADAPTERS
    from training.edit_dataset import validate_dataset
    data=Path(data);pending=[];manifest={};dataset_report={}
    if not (data/'manifest.json').is_file():pending.append('versioned SFT/whole-plan dataset manifest missing')
    else:
        manifest=json.loads((data/'manifest.json').read_text())
        try:
            if manifest.get('dataset_schema')=='legacy-vlm-export/2':dataset_report=_validate_legacy_tasks(data)
            else:dataset_report=validate_dataset(data)
        except (ValueError,KeyError,TypeError,OSError) as exc:dataset_report={'ok':False,'error':str(exc)}
        if not dataset_report.get('ok'):pending.append('dataset must pass exact runtime/provenance/family validation')
        if manifest.get('human_rows',0)<=0:pending.append('reviewed explicit-human demonstrations/preferences unavailable')
        if not (data/'split_manifest.json').is_file() and not (data/'family-splits.json').is_file():pending.append('frozen family split manifest missing')
    bundle=check_bundle(sft_bundle or '',verify_hashes=True)
    if bundle['status']!='ok':pending.append('compatible hash-verified SFT bundle required: '+bundle['status'])
    else:
        model=bundle['manifest'];registered=ADAPTERS[model['adapter_id']]
        if registered.verification=='ci_fixture':pending.append('random tiny CI fixture is not a real SFT policy')
        if not model.get('base',{}).get('revision') and not model.get('base',{}).get('file_sha256'):pending.append('base model revision/bytes identity must be frozen')
        if model.get('promotion',{}).get('independent_quality_verified') is not True:pending.append('SFT independent quality gate not certified')
    evidence=evidence or {}
    artifacts={}
    for name in ('human_preferences','reward_calibration','independent_evaluation','rejection_distillation','multimodal_grpo_compatibility'):
        path=Path(evidence.get(name,''))
        if not path.is_file():pending.append(name+' artifact missing');continue
        artifacts[name]={'path':str(path),'sha256':_hash(path)}
        try:document=json.loads(path.read_text()) if path.suffix=='.json' else None
        except ValueError:document=None
        if name=='human_preferences':
            try:
                rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
                if not any(r.get('provenance') in ('human','explicit_human') and (r.get('reviewer_id') or (r.get('review') or {}).get('reviewer_id')) for r in rows):pending.append('human preference artifact has no explicitly reviewed human records')
            except (ValueError,AttributeError):pending.append('human preference artifact invalid')
        elif name=='reward_calibration' and (not document or document.get('calibrated_against_humans') is not True):pending.append('reward human calibration incomplete')
        elif name=='independent_evaluation' and (not document or document.get('reward_version')==version_identity()['reward_version'] or not document.get('independent_human_review')):pending.append('independent evaluator/human holdout incomplete')
        elif name=='multimodal_grpo_compatibility' and (not document or document.get('finite_loss') is not True or document.get('completion_only_mask') is not True or document.get('frozen_reference_verified') is not True or document.get('adapter_id')!=(bundle.get('manifest') or {}).get('adapter_id')):pending.append('registered multimodal GRPO compatibility not verified')
        elif name=='rejection_distillation' and (not document or document.get('evaluated') is not True):pending.append('rejection-distillation comparison not evaluated')
    return {'schema':'grpo-prerequisites/1','ready':not pending,'status':'ready' if not pending else 'conditional_not_ready','pending':pending,
            'dataset':dataset_report,'data_manifest_sha256':_hash(data/'manifest.json') if (data/'manifest.json').is_file() else None,
            'sft_bundle_status':bundle['status'],'evidence':artifacts,'config':asdict(config),'reward_identity':version_identity(),
            'training_performed':False,'quality_improvement_claim':False}


def kernel_smoke():
    import torch
    torch.manual_seed(7);config=GRPOConfig();logp=torch.tensor([[-.7,-.2],[-.3,-.8],[-.4,-.6],[-.2,-.9]],requires_grad=True)
    old=logp.detach().clone();ref=old-.01;mask=torch.tensor([[1.,1.],[1.,0.],[1.,1.],[1.,1.]])
    advantages=torch.tensor(group_advantages([1.,.4,0.,-1.]))
    loss,stats=grpo_loss(logp,old,ref,advantages,mask,config);loss.backward()
    equal=group_advantages([-.1]*4);zero_loss,_=grpo_loss(old,old,old,torch.tensor(equal),mask,config)
    assert torch.isfinite(logp.grad).all() and logp.grad.abs().sum()>0 and float(zero_loss)==0
    return {'status':'finite_kernel_smoke','loss':float(loss.detach()),'stats':stats,'finite_gradients':bool(torch.isfinite(logp.grad).all()),
            'nonzero_gradient':bool(logp.grad.abs().sum()>0),'equal_reward_advantages':equal,'equal_reward_loss':float(zero_loss),
            'model_training_performed':False,'quality_improvement_claim':False,'torch':torch.__version__}


def train(data,sft_bundle,out,*,evidence=None,config=GRPOConfig(),device='cpu'):
    """Actual registered multimodal policy optimization, gated by independent evidence."""
    report=prerequisites(data,sft_bundle,evidence,config);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'preflight.json').write_text(json.dumps(report,indent=2))
    if not report['ready']:return report
    return _train_registered(data,sft_bundle,out,report,config,device)


def _train_registered(data,sft_bundle,out,preflight,config,device):
    """One on-policy group per update. Truncations/token mismatches never train."""
    import torch
    from training.edit_dpo import load_sft_policy,reference_fingerprint
    from training.edit_dataset import load_rows
    from training.vlm_adapters import _load_images,encode_example,collate,_to,export_bundle,IGNORE,gradient_report
    from engine.vlm import user_message,image_transform
    torch.manual_seed(config.seed);model,processor,adapter,bundle=load_sft_policy(sft_bundle,device)
    model.eval();dtype=bundle.get('dtype','float32');optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.lr)
    rows=_task_rows(data,'train');history=[];initial_reference=reference_fingerprint(model);steps=0
    def logps(example):
        batch=_to(collate([example],processor.tokenizer.pad_token_id or 0),device,dtype);labels=batch.pop('labels')[0]
        first=int((labels!=IGNORE).nonzero()[0]);length=labels.shape[0];logits=model(**batch,logits_to_keep=length-first+1).logits[0,:-1].float()
        return torch.log_softmax(logits,-1).gather(-1,labels[first:].clamp(min=0).unsqueeze(-1)).squeeze(-1)
    for index,row in enumerate(rows):
        if steps>=config.max_steps:break
        task=json.loads(row['task']);images=_load_images(row,data)
        text=processor.apply_chat_template([user_message(row['prompt'],len(images))],tokenize=False,add_generation_prompt=True)
        batch=processor(text=[text],images=images,return_tensors='pt').to(device);completions=[];examples=[];rejects=[]
        vision=[processor.tokenizer.convert_tokens_to_ids(token) for token in processor.tokenizer.all_special_tokens if any(key in token for key in ('image','vision','video'))]
        vision=[token for token in vision if token is not None and token!=processor.tokenizer.unk_token_id]
        with torch.no_grad():
            for _ in range(config.group_size):
                generated=model.generate(**batch,max_new_tokens=config.max_new_tokens,do_sample=True,temperature=config.temperature,top_p=config.top_p,suppress_tokens=vision or None)[0,batch['input_ids'].shape[1]:]
                eos=processor.tokenizer.eos_token_id
                if not len(generated) or int(generated[-1])!=eos:rejects.append('truncated_completion');continue
                completion=processor.tokenizer.decode(generated,skip_special_tokens=True)
                example=encode_example(processor,images,row['prompt'],completion)
                supervised=example['labels'][example['labels']!=IGNORE]
                if supervised.tolist()!=generated.tolist():rejects.append('generation_supervision_token_mismatch');continue
                completions.append(completion);examples.append(example)
        record=rollout(task['source'],task['target'],completions,locks=row.get('locks'),config=config,group_policies=(task.get('options') or {}).get('group_policies'))
        record.update(row_id=row['row_id'],rejected_generations=rejects,generated_tokens=sum(int((x['labels']!=IGNORE).sum()) for x in examples))
        if len(examples)==config.group_size and not record['zero_reward_variance']:
            with torch.no_grad():old=[logps(ex).detach() for ex in examples]
            model.set_adapter('ref')
            with torch.no_grad():reference=[logps(ex).detach() for ex in examples]
            model.set_adapter('default')
            for name,param in model.named_parameters():
                if '.ref.' in name:param.requires_grad_(False)
            losses=[]
            for ex,oldlp,reflp,adv in zip(examples,old,reference,record['advantages']):
                current=logps(ex);loss,stats=grpo_loss(current[None],oldlp[None],reflp[None],torch.tensor([adv],device=device),torch.ones_like(current)[None],config)
                losses.append(loss)
            loss=torch.stack(losses).mean();loss.backward();record['loss']=float(loss.detach());record['optimization_stats']=stats;record['gradients']=gradient_report(model)
            if not record['gradients']['grad_finite'] or record['gradients']['grad_non_lora']:raise ValueError('Unsafe GRPO gradients')
            optimizer.step();optimizer.zero_grad();steps+=1
        history.append(record)
        with (out/'rollouts.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    if reference_fingerprint(model)!=initial_reference:raise ValueError('Frozen SFT reference changed')
    status='trained_candidate' if steps else 'no_optimizer_steps'
    if steps:
        model.delete_adapter('ref');export_bundle(out/'bundle',model,processor,adapter,bundle['plan_schema'],tasks=['plan'],dtype=dtype,
           train_config={'stage':'grpo','config':asdict(config),'preflight':preflight,'reference_fingerprint':initial_reference},status='trained_grpo_candidate')
    result={'status':status,'optimizer_steps':steps,'groups':len(history),'training_performed':steps>0,'quality_improvement_claim':False,'config':asdict(config),'reward_identity':version_identity()}
    (out/'result.json').write_text(json.dumps(result,indent=2));return result

EXPERIMENT_SPEC={'schema':'grpo-experiment/1','arms':['frozen validated SFT','rejection-sampling/distillation','whole-plan GRPO'],
 'selection':'independent family-held-out human preference and structural/render validity; reward-only improvement insufficient',
 'equal_inference_budget':['same source/target/family split','sample count','decoding and max tokens','repair/solver/render budgets'],
 'monitor':['raw and repaired validity separately','invalid/hallucinated IDs and locks','reward channels','repair burden','truncation/token mismatch','entropy and KL','zero-variance groups','renderer/model calls','whole-process cost'],
 'rollback':'exact frozen SFT bundle; learned providers stay optional',
 'sequential_edit':'deferred unless coordinated-edit failure is demonstrated; use imitation and deterministic beam/LNS baseline first',
 'training_performed':False,'quality_improvement_claim':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['dry-run','kernel-smoke','experiment-spec','train'])
    parser.add_argument('--data',type=Path,default=Path('output/learn/data'));parser.add_argument('--sft-bundle',type=Path);parser.add_argument('--out',type=Path,default=Path('output/training/grpo-audit'))
    parser.add_argument('--evidence',type=Path);parser.add_argument('--device',default='cpu');parser.add_argument('--max-steps',type=int,default=20)
    args=parser.parse_args();evidence=json.loads(args.evidence.read_text()) if args.evidence else {};config=GRPOConfig(max_steps=args.max_steps)
    if args.command=='kernel-smoke':result=kernel_smoke()
    elif args.command=='experiment-spec':result=EXPERIMENT_SPEC
    elif args.command=='dry-run':result=prerequisites(args.data,args.sft_bundle,evidence,config)
    else:result=train(args.data,args.sft_bundle,args.out,evidence=evidence,config=config,device=args.device)
    print(json.dumps(result,indent=2))
