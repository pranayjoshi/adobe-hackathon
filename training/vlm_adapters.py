"""Training-side model adapters: supervision masking, LoRA SFT on MPS/CPU, bundle export, benchmark.

The adapter *registry* is runtime-side in ``engine/vlm.py`` (ADAPTERS) so the engine can check a
bundle without ML imports; this module consumes it. Nothing here downloads: every load uses
``local_files_only=True``.

  python -m training.vlm_adapters sweep-spec
  python -m training.vlm_adapters smoke --data DATA --out OUT [--adapter tiny-qwen2.5-vl-test] [--device cpu|mps]
  python -m training.vlm_adapters benchmark --data DATA --out OUT [--bundle B ...] [--limit N]

Interfaces later phases (DPO 16, RL 17) build on:
  encode_example(processor, images, prompt, completion) -> dict      # labels supervise completion + end-of-turn only
  supervised_text(processor, example) -> str
  collate(examples, pad_id) -> batch
  lora_config(adapter, rank, alpha=None, dropout=.05) -> peft.LoraConfig
  gradient_report(model) -> dict                                      # non-LoRA gradients are an error
  train_sft(train_rows, val_rows, data, adapter_id, out, **cfg) -> dict
  export_bundle(out, model, processor, adapter, plan_schema, ...) -> manifest   # engine.vlm.check_bundle-compatible
  evaluate_policy(policy, rows, data, limit) -> dict                  # parse / raw-valid / repaired-valid / latency
"""
import argparse,hashlib,json,math,os,platform,resource,statistics,sys,time
os.environ.setdefault('HF_HUB_OFFLINE','1')  # never reach the network, even on a cache miss
from pathlib import Path
from engine.vlm import ADAPTERS,BUNDLE_SCHEMA,TransformersPolicy,adapter as get_adapter,check_bundle,image_transform,user_message

IGNORE=-100
SWEEP={'method':'LoRA (PEFT), language-model projections only; vision tower frozen',
       'grid':[{'rank':r,'alpha':2*r,'lr':lr} for r in (8,16,32) for lr in (5e-5,1e-4)],
       'dropout':.05,'effective_batch':8,'per_device_batch':1,'grad_accum':8,'max_epochs':3,'early_stop':{'metric':'validation raw_valid_rate then loss','patience_evals':2},
       'selection':'held-out raw-plan validity (validate_plan + exact materialization), then end-to-end designer preference; never token loss alone',
       'precision':{'mps':'bfloat16 weights, float32 LoRA params','cpu':'float32'},'quantization':'none for training (no bitsandbytes on MPS); quantize only after unquantized gains'}


class MaskingError(AssertionError):
    """The chat template/tokenizer does not allow exact assistant-only supervision."""


def select_device(prefer='auto'):
    import torch
    if prefer not in ('auto','cpu','mps','cuda'):raise ValueError('device must be auto|cpu|mps|cuda')
    if prefer=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable')
    if prefer=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if prefer!='auto':return prefer
    return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def dtype_name(device,a,prefer='auto'):
    if prefer!='auto':return prefer
    # Tiny fixtures stay float32 for exact, deterministic smoke checks.
    return 'float32' if device=='cpu' or a.verification=='ci_fixture' else 'bfloat16'


def load_base(a,base=None,device='cpu',dtype='float32'):
    import torch,transformers
    from transformers import AutoProcessor
    if not base and not a.revision:raise ValueError('Pin the registered base revision or provide a byte-identified local base before training')
    src=base or a.hf_repo
    kw={'local_files_only':True}
    if not base and a.revision:kw['revision']=a.revision
    processor=AutoProcessor.from_pretrained(src,**kw)
    cls=getattr(transformers,a.model_class,None) or transformers.AutoModelForImageTextToText
    model=cls.from_pretrained(src,dtype=getattr(torch,dtype),**kw).to(device)
    return model,processor


def _load_images(row,data):
    from PIL import Image
    t=image_transform('letterbox_white_672');out=[]
    for rel in row['images']:
        with Image.open(Path(data)/rel) as im:out.append(t(im))
    return out


def encode_example(processor,images,prompt,completion):
    """Tokenize one conversation; labels supervise exactly ``completion`` + the end-of-turn token."""
    tok=processor.tokenizer
    user=user_message(prompt,len(images))
    prompt_text=processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True)
    full_text=processor.apply_chat_template([user,{'role':'assistant','content':[{'type':'text','text':completion}]}],tokenize=False)
    if not full_text.startswith(prompt_text):raise MaskingError('chat template: prompt is not a prefix of the conversation')
    tail=full_text[len(prompt_text):]
    if not tail.startswith(completion):raise MaskingError('chat template rewrites the assistant completion')
    end=tail[len(completion):]
    eot=end.strip()  # e.g. <|im_end|>; trailing newline/whitespace after it is not supervised
    full=processor(text=[full_text],images=images or None,return_tensors='pt')
    head=processor(text=[prompt_text],images=images or None,return_tensors='pt')
    ids=full['input_ids'][0];P=head['input_ids'].shape[1]
    if not bool((ids[:P]==head['input_ids'][0]).all()):raise MaskingError('prompt tokens differ inside the full conversation')
    sup=tok(completion+eot,add_special_tokens=False)['input_ids']
    if ids[P:P+len(sup)].tolist()!=sup:raise MaskingError('completion tokens merge across the prompt boundary')
    labels=ids.clone().fill_(IGNORE);labels[P:P+len(sup)]=ids[P:P+len(sup)]
    ex={k:v[0] if k in ('input_ids','attention_mask','mm_token_type_ids','token_type_ids') else v for k,v in full.items()}
    ex['labels']=labels;ex['_expected']=completion+eot
    return ex


def supervised_text(processor,ex):
    lab=ex['labels'];return processor.tokenizer.decode(lab[lab!=IGNORE].tolist(),skip_special_tokens=False)


def assert_supervision(processor,ex):
    got=supervised_text(processor,ex)
    if got!=ex['_expected']:raise MaskingError('supervised tokens != assistant plan: %r'%got[:80])
    image_ids={i for i in (processor.tokenizer.convert_tokens_to_ids(t) for t in ('<|image_pad|>','<|vision_start|>','<|vision_end|>')) if isinstance(i,int) and i>=0}
    if any(int(t) in image_ids for t in ex['labels'][ex['labels']!=IGNORE]):raise MaskingError('image tokens are supervised')
    return True


def collate(examples,pad_id):
    """Right-pad a list of encode_example outputs; images concatenated in order."""
    import torch
    L=max(e['input_ids'].shape[0] for e in examples);out={}
    def pad(key,value):
        return torch.stack([torch.nn.functional.pad(e[key],(0,L-e[key].shape[0]),value=value) for e in examples])
    out['input_ids']=pad('input_ids',pad_id);out['attention_mask']=pad('attention_mask',0);out['labels']=pad('labels',IGNORE)
    for k in ('mm_token_type_ids','token_type_ids'):
        if k in examples[0]:out[k]=pad(k,0)
    for k in ('pixel_values','image_grid_thw'):
        if k in examples[0]:out[k]=torch.cat([e[k] for e in examples])
    return out


def lora_config(a,rank=16,alpha=None,dropout=.05):
    from peft import LoraConfig
    return LoraConfig(r=rank,lora_alpha=alpha or 2*rank,lora_dropout=dropout,target_modules=a.lora_targets,task_type='CAUSAL_LM')


def gradient_report(model):
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    with_grad=[n for n,p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0]
    return {'trainable':len(trainable),'trainable_non_lora':[n for n in trainable if 'lora_' not in n][:10],
            'with_grad':len(with_grad),'grad_non_lora':[n for n in with_grad if 'lora_' not in n][:10],
            'grad_finite':all(math.isfinite(float(p.grad.abs().sum())) for _,p in model.named_parameters() if p.grad is not None),
            'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_params':sum(p.numel() for p in model.parameters())}


def _to(batch,device,dtype):
    import torch
    return {k:(v.to(device,dtype=getattr(torch,dtype)) if k=='pixel_values' else v.to(device)) for k,v in batch.items()}


def peak_rss_gb():
    r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r/(1<<30) if sys.platform=='darwin' else r/(1<<20),3)


def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def environment():
    import torch,transformers,peft
    try:import trl;tv=trl.__version__
    except ImportError:tv=None
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,
            'transformers':transformers.__version__,'peft':peft.__version__,'trl':tv,'mps':torch.backends.mps.is_available(),'cuda':torch.cuda.is_available()}


def _base_identity(a,base):
    if base:
        base=Path(base).resolve();files={p.relative_to(base).as_posix():_sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p.suffix in ('.json','.safetensors','.jinja','.txt','.bin')}
        return {'repo':a.hf_repo,'local_path':str(base),'revision':None,'file_sha256':files}
    try:
        from huggingface_hub import snapshot_download
        path=snapshot_download(a.hf_repo,revision=a.revision,local_files_only=True)
        files={p.name:_sha(p) for p in sorted(Path(path).iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors','.jinja','.txt')}
    except Exception as exc:
        path,files=None,{'error':type(exc).__name__}
    return {'repo':a.hf_repo,'revision':a.revision,'snapshot':path,'file_sha256':files}


def export_bundle(out,model,processor,a,plan_schema,*,tasks,base=None,dtype='float32',dataset_manifest=None,train_config=None,metrics=None,status='trained'):
    """Adapter + processor/chat template + schema + preprocessing + dataset/training/env hashes."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(str(out/'adapter'));processor.save_pretrained(str(out/'processor'))
    files={p.relative_to(out).as_posix():_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='bundle_manifest.json'}
    tok=out/'processor'/'tokenizer.json'
    manifest={'bundle_schema':BUNDLE_SCHEMA,'adapter_id':a.adapter_id,'adapter':a.to_dict(),'base':_base_identity(a,base),'dtype':dtype,
              'plan_schema':plan_schema,'tasks':list(tasks),'preprocessing':{'image_transform':a.image_transform,'image_size':a.image_size,'render':'renderer.converter.convert.render_design_to_png(max_size=672)'},
              'chat_template_sha256':hashlib.sha256((processor.chat_template or '').encode()).hexdigest(),
              'tokenizer_sha256':_sha(tok) if tok.exists() else None,
              'dataset_manifest_sha256':hashlib.sha256(json.dumps(dataset_manifest,sort_keys=True).encode()).hexdigest() if dataset_manifest else None,
              'dataset_counts':(dataset_manifest or {}).get('counts'),'dataset_provenance':(dataset_manifest or {}).get('provenance'),
              'training':train_config,'training_sha256':hashlib.sha256(json.dumps(train_config,sort_keys=True,default=str).encode()).hexdigest() if train_config else None,
              'environment':environment(),'metrics':metrics,'status':status,'published':False,
              'deployment':'transformers backend only (CHAMELEON_VLM_BACKEND=transformers CHAMELEON_VLM_BUNDLE=<dir>); llama.cpp needs a separately verified merge+GGUF conversion',
              'files':files}
    (out/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+'\n')
    return manifest


def _examples(rows,data,processor):
    out=[]
    for r in rows:
        ex=encode_example(processor,_load_images(r,data),r['prompt'],r['completion']);assert_supervision(processor,ex);out.append(ex)
    return out


def _loss(model,batch):
    out=model(**{k:v for k,v in batch.items()})
    return out.loss


def train_sft(train_rows,val_rows,data,adapter_id,out,*,base=None,rank=16,lr=1e-4,max_steps=20,epochs=1,accum=1,eval_every=5,patience=2,device='auto',dtype='auto',seed=7,dataset_manifest=None,max_new_tokens=256,gen_limit=2):
    """LoRA SFT with a transparent loop (no CUDA/bitsandbytes assumptions). Returns a typed result."""
    import torch
    from peft import get_peft_model
    torch.manual_seed(seed)
    a=get_adapter(adapter_id);device=select_device(device);dtype=dtype_name(device,a,dtype);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if isinstance(accum,bool) or not isinstance(accum,int) or accum<1:raise ValueError('accum must be positive integer')
    started=time.monotonic();model,processor=load_base(a,base,device,dtype)
    model=get_peft_model(model,lora_config(a,rank));model.train()
    pad=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    train_ex=_examples(train_rows,data,processor);val_ex=_examples(val_rows,data,processor)
    if not train_ex:return {'status':'no_training_rows'}
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.0)
    log=(out/'train_log.jsonl').open('w');history=[];grad=None;best=(math.inf,None);best_state=None;bad=0;step=0;order=list(range(len(train_ex)))
    def val_loss():
        if not val_ex:return None
        model.eval();tot=0.0
        with torch.no_grad():
            for e in val_ex:tot+=float(_loss(model,_to(collate([e],pad),device,dtype)))
        model.train();return tot/len(val_ex)
    status='trained'
    import random
    rng=random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        for n,i in enumerate(order):
            loss=_loss(model,_to(collate([train_ex[i]],pad),device,dtype))/accum
            if not torch.isfinite(loss):status='non_finite_loss';break
            loss.backward()
            if (n+1)%accum and n+1!=len(order):continue
            pending=(n%accum)+1
            if pending<accum:
                for param in params:
                    if param.grad is not None:param.grad.mul_(accum/pending)
            if grad is None:
                grad=gradient_report(model)
                if grad['grad_non_lora'] or grad['trainable_non_lora'] or not grad['grad_finite'] or grad['with_grad']==0:status='bad_gradients';break
            opt.step();opt.zero_grad();step+=1
            rec={'step':step,'epoch':epoch,'loss':round(float(loss)*accum,6)}
            if val_ex and step%eval_every==0:
                rec['val_loss']=round(val_loss(),6)
                if rec['val_loss']<best[0]-1e-6:
                    best=(rec['val_loss'],step);best_state={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};bad=0
                else:bad+=1
            history.append(rec);log.write(json.dumps(rec)+'\n');log.flush()
            if step>=max_steps or bad>=patience:break
        if status!='trained' or step>=max_steps or bad>=patience:break
    log.close()
    if step==0 and status=='trained':status='no_optimizer_steps'
    if best_state is not None:
        with torch.no_grad():
            for name,param in model.named_parameters():
                if name in best_state:param.copy_(best_state[name])
    vl=val_loss() if status=='trained' else None
    config={'adapter_id':adapter_id,'base':str(base) if base else a.hf_repo,'revision':a.revision,'rank':rank,'alpha':2*rank,'dropout':.05,'targets':a.lora_targets,
            'lr':lr,'max_steps':max_steps,'epochs':epochs,'accum':accum,'seed':seed,'device':device,'dtype':dtype,'steps_done':step,
            'train_rows':len(train_rows),'val_rows':len(val_rows),'row_ids':sorted(r['row_id'] for r in train_rows+val_rows if 'row_id' in r)}
    result={'status':status,'steps':step,'history':history,'final_val_loss':vl,'gradients':grad,'seconds':round(time.monotonic()-started,2),
            'peak_rss_gb':peak_rss_gb(),'mps_allocated_gb':round(torch.mps.driver_allocated_memory()/(1<<30),3) if device=='mps' else None,'config':config}
    if status!='trained':return result
    plan_schema=(dataset_manifest or {}).get('plan_schema','layout-plan/2');kinds=sorted({r['kind'] for r in train_rows})
    bundle=out/'bundle'
    export_bundle(bundle,model,processor,a,plan_schema,tasks=kinds,base=base,dtype=dtype,dataset_manifest=dataset_manifest,train_config=config,
                  metrics={'final_val_loss':vl,'train_loss_last':history[-1]['loss'] if history else None},status='trained_'+('smoke' if a.verification=='ci_fixture' else 'candidate'))
    result['bundle']=str(bundle);result['bundle_check']=check_bundle(bundle,verify_hashes=True)['status']
    return result


# ------------------------------------------------------------- evaluation

def evaluate_policy(policy,rows,data,limit=None,max_new_tokens=256):
    """Generate -> parse -> typed validation -> exact materialization -> engine repair."""
    from engine.analysis import analyze
    from engine.contracts import DesignRevision
    from engine.invariants import passes,validate
    from engine.materialize import apply_edit
    from engine.neural_proposals import PlanRejected,action_to_proposal,materialize_plan,parse_plan_text
    from engine.repair import repair
    from collections import Counter
    rec=[];reasons=Counter()
    for r in rows[:limit]:
        task=json.loads(r['task']);A=analyze(task['source'],group_policies=(task.get('options') or {}).get('group_policies'));locks=r.get('locks') or {}
        try:text,stats=policy.generate(_load_images(r,data),r['prompt'],max_new_tokens)
        except Exception as exc:rec.append({'row_id':r['row_id'],'status':'generation_error','detail':type(exc).__name__});reasons['generation_error']+=1;continue
        plan=parse_plan_text(text);item={'row_id':r['row_id'],'kind':r['kind'],'seconds':stats['seconds'],'new_tokens':stats['new_tokens'],'reply_head':text[:80]}
        if plan is None:item.update(status='unparseable',raw_valid=False,repaired_valid=False);reasons['unparseable']+=1;rec.append(item);continue
        if r['kind']=='edit':
            rev=DesignRevision.create(task['state'])
            try:apply_edit(task['source'],rev,action_to_proposal(plan,rev,provenance='model',locks=locks),locks);item.update(status='ok',raw_valid=True,repaired_valid=True)
            except PlanRejected as exc:item.update(status='rejected',raw_valid=False,repaired_valid=False,reasons=[x['code'] for x in exc.reasons][:8]);reasons.update(x['code'] for x in exc.reasons)
            except ValueError as exc:item.update(status='materialization_rejected',raw_valid=False,repaired_valid=False);reasons['materialization_rejected']+=1
            rec.append(item);continue
        o=materialize_plan(task['source'],task['target'],A,plan,locks)
        item.update(status=o.status,raw_valid=o.status=='ok' and o.raw_valid,reasons=[x['code'] for x in o.reasons][:8])
        reasons.update(x['code'] for x in o.reasons) if o.reasons else None
        if o.status=='ok':
            fixed,_=repair(task['source'],o.design,A);item['repaired_valid']=passes(validate(task['source'],fixed))
        else:item['repaired_valid']=False
        rec.append(item)
    secs=[x['seconds'] for x in rec if 'seconds' in x]
    n=len(rec)
    return {'tasks':n,'parse_rate':round(sum(x['status']!='unparseable' and x['status']!='generation_error' for x in rec)/n,4) if n else None,
            'raw_valid_rate':round(sum(bool(x.get('raw_valid')) for x in rec)/n,4) if n else None,
            'repaired_valid_rate':round(sum(bool(x.get('repaired_valid')) for x in rec)/n,4) if n else None,
            'latency_s':{'median':round(statistics.median(secs),3),'p95':round(sorted(secs)[max(0,math.ceil(.95*len(secs))-1)],3)} if secs else None,
            'reasons':dict(reasons.most_common()),'records':rec,'crashed':False}


class BaseModelPolicy:
    """Untuned base model through the same prompt/transform path (benchmark arm)."""
    def __init__(self,adapter_id,device='auto'):
        self.adapter=get_adapter(adapter_id);self.device=select_device(device);dt=dtype_name(self.device,self.adapter)
        self.model,self.processor=load_base(self.adapter,None,self.device,dt);self.model.eval();self.dtype=dt
    generate=TransformersPolicy.generate


class ReferencePolicy:
    """Teacher completions through the identical validator: proves the harness, not a model."""
    def __init__(self,rows):self.by={r['prompt']:r['completion'] for r in rows}
    def generate(self,images,prompt,max_new_tokens=0):return self.by[prompt],{'seconds':0.0,'new_tokens':0}


BENCHMARK_SPEC={
    'benchmark':'compact-vlm-plan-proposals/1',
    'tasks':'held-out validation-split rows of a training.edit_dataset build (non-suite families only; frozen split manifest); plan and edit kinds reported separately',
    'arms':[{'id':'reference_teacher','what':'dataset completions through the same validator (harness check, upper bound for raw validity)'},
            {'id':'base:<adapter_id>','what':'untuned base model, identical prompt/transform'},
            {'id':'sft:<bundle>','what':'exported LoRA bundle loaded through engine.vlm.TransformersPolicy (runtime path)'}],
    'whole_plan_baseline':'qwen2.5-vl-3b-instruct with layout-plan/1 prompt (current runtime contract); deterministic engine output is the teacher reference',
    'metrics':['parse_rate','raw_valid_rate (typed validation + exact materialization, no containment shift)','repaired_valid_rate (after engine.repair.repair, invariants pass)',
               'typed rejection reasons','latency median/p95 per proposal','peak RSS GB','MPS driver memory GB'],
    'decoding':{'do_sample':False,'max_new_tokens':768},
    'promotion':'unique valid candidate wins at the selected resource profile in the end-to-end pipeline + family-clustered designer preference; this benchmark alone never promotes',
}


def benchmark(data,out,bundles=(),adapters=None,limit=None,device='auto',max_new_tokens=256):
    from training.edit_dataset import load_rows
    import torch
    rows=load_rows(data,'plan','validation')+load_rows(data,'edit','validation')
    if not rows:rows=load_rows(data,'plan','train')[:limit or 4];held='train_fallback_no_validation_rows'
    else:held='validation'
    report={'spec':BENCHMARK_SPEC,'data':str(data),'task_split':held,'environment':environment(),'arms':{}}
    report['arms']['reference_teacher']=_strip(evaluate_policy(ReferencePolicy(rows),rows,data,limit))
    for aid in (adapters if adapters is not None else list(ADAPTERS)):
        a=ADAPTERS[aid]
        if 'transformers' not in a.backends:report['arms']['base:'+aid]={'status':'pending','prerequisite':'; '.join(a.prerequisites) or 'no transformers backend'};continue
        try:
            t=time.monotonic();pol=BaseModelPolicy(aid,device);load=round(time.monotonic()-t,2)
        except Exception as exc:
            report['arms']['base:'+aid]={'status':'pending','prerequisite':'; '.join(a.prerequisites) or type(exc).__name__,'load_error':type(exc).__name__+': '+str(exc)[:160]};continue
        r=_strip(evaluate_policy(pol,rows,data,limit,max_new_tokens));r.update(status='ran',load_seconds=load,peak_rss_gb=peak_rss_gb(),verification=a.verification)
        report['arms']['base:'+aid]=r;del pol
    for b in bundles:
        chk=check_bundle(b,verify_hashes=True)
        if chk['status']!='ok':report['arms']['sft:'+str(b)]={'status':'fail_closed','bundle_status':chk['status'],'reasons':chk['reasons']};continue
        t=time.monotonic();pol=TransformersPolicy.load(b,select_device(device));load=round(time.monotonic()-t,2)
        r=_strip(evaluate_policy(pol,rows,data,limit,max_new_tokens));r.update(status='ran',load_seconds=load,peak_rss_gb=peak_rss_gb(),
              mps_allocated_gb=round(torch.mps.driver_allocated_memory()/(1<<30),3) if pol.device=='mps' else None,adapter_id=chk['adapter_id'])
        report['arms']['sft:'+str(b)]=r
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n')
    return report


def _strip(r):
    r=dict(r);r['records']=r['records'][:12];return r


def smoke(data,out,adapter_id='tiny-qwen2.5-vl-test',device='auto',max_steps=6,rank=8,lr=1e-4):
    """Real short LoRA SFT on a locally cached model, export, reload through the runtime adapter, generate."""
    from training.edit_dataset import load_rows
    manifest=json.loads((Path(data)/'manifest.json').read_text())
    train=load_rows(data,'plan','train')[:4]+load_rows(data,'edit','train')[:4]
    val=(load_rows(data,'plan','validation')+load_rows(data,'edit','validation'))[:2]
    res=train_sft(train,val,data,adapter_id,out,rank=rank,lr=lr,max_steps=max_steps,eval_every=2,device=device,dataset_manifest=manifest)
    if res.get('bundle'):
        pol=TransformersPolicy.load(res['bundle'],select_device(device))
        res['reload_generation']=_strip(evaluate_policy(pol,(val or train)[:2]+train[:1],data,None,64))
    (Path(out)/'smoke_result.json').write_text(json.dumps(res,indent=2,default=str)+'\n')
    return res


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('sweep-spec');sub.add_parser('benchmark-spec');sub.add_parser('registry')
    s=sub.add_parser('smoke');s.add_argument('--data',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    s.add_argument('--adapter',default='tiny-qwen2.5-vl-test');s.add_argument('--device',default='auto');s.add_argument('--max-steps',type=int,default=6)
    b=sub.add_parser('benchmark');b.add_argument('--data',type=Path,required=True);b.add_argument('--out',type=Path,required=True)
    b.add_argument('--bundle',type=Path,action='append',default=[]);b.add_argument('--adapter',action='append');b.add_argument('--limit',type=int);b.add_argument('--device',default='auto')
    b.add_argument('--max-new-tokens',type=int,default=256)
    a=ap.parse_args()
    if a.cmd=='sweep-spec':print(json.dumps(SWEEP,indent=2))
    elif a.cmd=='benchmark-spec':print(json.dumps(BENCHMARK_SPEC,indent=2))
    elif a.cmd=='registry':print(json.dumps({k:v.to_dict() for k,v in ADAPTERS.items()},indent=2))
    elif a.cmd=='smoke':
        r=smoke(a.data,a.out,a.adapter,a.device,a.max_steps);print(json.dumps({k:v for k,v in r.items() if k!='history'},indent=2,default=str));raise SystemExit(0 if r['status']=='trained' else 1)
    else:
        r=benchmark(a.data,a.out,a.bundle,a.adapter,a.limit,a.device,a.max_new_tokens)
        print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='records'} for k,v in r['arms'].items()},indent=2,default=str))
