"""Optional local visual proposals. Never downloads weights or calls a provider.

Two runtime backends, both off by default (``CHAMELEON_VLM=1`` enables):

* ``llama_cpp`` (legacy default): ``CHAMELEON_VLM_MODEL`` GGUF + ``CHAMELEON_VLM_MMPROJ`` projector.
* ``transformers``: ``CHAMELEON_VLM_BACKEND=transformers`` + ``CHAMELEON_VLM_BUNDLE=<exported bundle dir>``
  (``training.vlm_adapters.export_bundle``). The bundle names a registered adapter;
  anything incompatible fails closed with a typed ``bundle_status`` and the
  deterministic engine continues unchanged.

The model adapter registry lives here (runtime side, no ML imports) so training
and inference share one definition of model class, chat template, image
transform and plan schema. ``training/vlm_adapters.py`` holds the training-side
collator, trainer, bundle export and benchmark.
"""
import base64,hashlib,importlib.util,io,json,os,time
from dataclasses import asdict,dataclass,field
from pathlib import Path
_MODEL=None
_POLICY=None

BUNDLE_SCHEMA='vlm-bundle/1'
BACKENDS=('llama_cpp','transformers')
LM_LORA_TARGETS=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'


@dataclass(frozen=True)
class ModelAdapter:
    """Everything that must agree between training and inference for one model family."""
    adapter_id:str
    hf_repo:str
    revision:str|None              # pinned snapshot; None = unpinned (must be pinned before a real run)
    model_class:str                # transformers class name
    processor_class:str
    chat_template:str              # 'processor' = the processor's bundled template (saved into every bundle)
    image_transform:str='letterbox_white_672'
    image_size:int=672
    plan_schemas:tuple=('layout-plan/1','layout-plan/2','edit-action/1')
    backends:tuple=('transformers',)
    tokenizer_version:str|None=None   # tokenizer file hash recorded at export; None until verified
    min_transformers:str='4.49'
    lora_targets:str=LM_LORA_TARGETS
    resource_envelope:dict=field(default_factory=dict)
    verification:str='declared_unverified'   # ci_fixture | baseline_unverified | declared_unverified | smoke_verified
    prerequisites:tuple=()            # exact unmet prerequisites on this machine, if known
    notes:str=''

    def to_dict(self):return asdict(self)


ADAPTERS={a.adapter_id:a for a in (
    ModelAdapter('tiny-qwen2.5-vl-test','trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration','62f13fe1c31492fb4195c72ec52cbb736f72fefb',
                 'Qwen2_5_VLForConditionalGeneration','Qwen2_5_VLProcessor','processor',min_transformers='5.0',
                 resource_envelope={'params':3.9e6,'device':'cpu|mps','train_peak_gb':.5},verification='ci_fixture',
                 notes='Random-weight test model. Pipeline checks only: its generations are garbage and must be rejected cleanly.'),
    ModelAdapter('qwen2.5-vl-3b-instruct','Qwen/Qwen2.5-VL-3B-Instruct',None,'Qwen2_5_VLForConditionalGeneration','Qwen2_5_VLProcessor','processor',
                 backends=('transformers','llama_cpp'),
                 resource_envelope={'params':3.75e9,'bf16_weights_gb':7.5,'lora_train_peak_gb_est':12,'device':'mps(bf16)|cpu(fp32, slow)','min_ram_gb':16},
                 verification='baseline_unverified',
                 prerequisites=('weights not in local HF cache (no network downloads allowed)','revision not pinned'),
                 notes='Current train_vlm default. llama_cpp use needs merged export -> GGUF + mmproj conversion (separately verified).'),
    ModelAdapter('qwen3-vl-4b-instruct','Qwen/Qwen3-VL-4B-Instruct',None,'Qwen3VLForConditionalGeneration','Qwen3VLProcessor','processor',min_transformers='4.57',
                 resource_envelope={'params':4.4e9,'bf16_weights_gb':8.9,'lora_train_peak_gb_est':14,'device':'mps(bf16)','min_ram_gb':16},
                 prerequisites=('weights not in local HF cache (no network downloads allowed)','revision not pinned','letterbox 672 vs 16px patch/2x merge unverified'),
                 notes='Declared, unverified. 672 is a multiple of 32, so the letterbox needs no processor resize if min/max pixels allow 672^2.'),
    ModelAdapter('qwen3.5-4b','Qwen/Qwen3.5-4B',None,'Qwen3_5ForConditionalGeneration','AutoProcessor','processor',min_transformers='5.0',
                 resource_envelope={'params':4.5e9,'bf16_weights_gb':9,'lora_train_peak_gb_est':15,'device':'mps(bf16)','min_ram_gb':16},
                 lora_targets=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|out_proj)$',
                 prerequisites=('HF safetensors weights not in local HF cache (no network downloads allowed)','revision not pinned','hybrid linear-attention LoRA targets unverified'),
                 notes='Declared, unverified. Loader is Qwen3_5ForConditionalGeneration, not the Qwen2.5 path.'),
    ModelAdapter('qwen3.5-4b-gguf-q4km','unsloth/Qwen3.5-4B-GGUF',None,'gguf','mmproj-F16.gguf','gguf-embedded',
                 backends=('llama_cpp',),plan_schemas=('layout-plan/1','layout-plan/2'),
                 resource_envelope={'file_gb':2.7,'device':'cpu(llama.cpp)'},
                 prerequisites=('llama-cpp-python not installed (no network installs allowed)',),
                 notes='Inference-only GGUF (Qwen3.5-4B-Q4_K_M.gguf + mmproj-F16.gguf are cached locally). Not trainable here; untuned base.'),
)}


def adapter(adapter_id):
    if adapter_id not in ADAPTERS:raise KeyError('unknown model adapter: '+str(adapter_id))
    return ADAPTERS[adapter_id]


def prepare_image(image,size=672):
    """Letterbox onto a fixed white square. Training (training/train_vlm.py) uses the same input."""
    from PIL import Image
    image=image.convert('RGB');image.thumbnail((size,size));canvas=Image.new('RGB',(size,size),'white')
    canvas.paste(image,((size-image.width)//2,(size-image.height)//2));return canvas


IMAGE_TRANSFORMS={'letterbox_white_672':lambda image:prepare_image(image,672)}


def image_transform(name):
    if name not in IMAGE_TRANSFORMS:raise KeyError('unknown image transform: '+str(name))
    return IMAGE_TRANSFORMS[name]


def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def check_bundle(path,verify_hashes=False):
    """Typed compatibility check of an exported bundle. Never imports torch.

    status: ok | missing_bundle | unreadable_manifest | wrong_bundle_schema | unknown_adapter |
            incompatible_transform | incompatible_schema | unsupported_backend | missing_file | hash_mismatch
    """
    if not path:return {'status':'missing_bundle','reasons':['CHAMELEON_VLM_BUNDLE not set']}
    root=Path(path);mf=root/'bundle_manifest.json'
    if not mf.is_file():return {'status':'missing_bundle','reasons':[str(mf)+' not found']}
    try:m=json.loads(mf.read_text())
    except ValueError:return {'status':'unreadable_manifest','reasons':['manifest is not JSON']}
    if not isinstance(m,dict) or m.get('bundle_schema')!=BUNDLE_SCHEMA:return {'status':'wrong_bundle_schema','reasons':[str(m.get('bundle_schema') if isinstance(m,dict) else None)]}
    aid=m.get('adapter_id')
    if aid not in ADAPTERS:return {'status':'unknown_adapter','reasons':[str(aid)]}
    a=ADAPTERS[aid];pre=m.get('preprocessing') or {}
    if pre.get('image_transform')!=a.image_transform or pre.get('image_size')!=a.image_size or pre.get('image_transform') not in IMAGE_TRANSFORMS:
        return {'status':'incompatible_transform','reasons':[str(pre)]}
    if m.get('plan_schema') not in a.plan_schemas:return {'status':'incompatible_schema','reasons':[str(m.get('plan_schema'))]}
    if 'transformers' not in a.backends:return {'status':'unsupported_backend','reasons':[aid+' has no transformers backend']}
    files=m.get('files') or {}
    for rel,digest in sorted(files.items()):
        if not isinstance(rel,str) or Path(rel).is_absolute() or not (root/rel).resolve().is_relative_to(root.resolve()):return {'status':'unsafe_bundle_path','reasons':[str(rel)]}
        f=root/rel
        if not f.is_file():return {'status':'missing_file','reasons':[rel]}
        if verify_hashes and _sha256(f)!=digest:return {'status':'hash_mismatch','reasons':[rel]}
    if not files:return {'status':'missing_file','reasons':['manifest lists no files']}
    if verify_hashes:
        base=m.get('base') or {};location=base.get('local_path') or base.get('snapshot')
        for rel,expected in (base.get('file_sha256') or {}).items():
            if not location or not isinstance(rel,str) or Path(rel).is_absolute() or '..' in Path(rel).parts:return {'status':'unsafe_base_path','reasons':[str(rel)]}
            path=Path(location)/rel
            if not path.is_file() or _sha256(path)!=expected:return {'status':'base_hash_mismatch','reasons':[str(rel)]}
    return {'status':'ok','reasons':[],'adapter_id':aid,'plan_schema':m['plan_schema'],'manifest':m}


def _has(module):
    try:return importlib.util.find_spec(module) is not None
    except (ValueError,ImportError):return False


def status():
    enabled=os.environ.get('CHAMELEON_VLM','0')=='1'
    backend=os.environ.get('CHAMELEON_VLM_BACKEND','llama_cpp')
    if backend=='transformers':
        check=check_bundle(os.environ.get('CHAMELEON_VLM_BUNDLE',''))
        return {'enabled':enabled,'backend':backend,'weights_present':check['status']=='ok','bundle_status':check['status'],
                'runtime_available':_has('torch') and _has('transformers') and _has('peft'),'plan_schema':check.get('plan_schema'),'adapter_id':check.get('adapter_id')}
    if backend!='llama_cpp':
        return {'enabled':enabled,'backend':backend,'weights_present':False,'bundle_status':'unsupported_backend','runtime_available':False}
    path=os.environ.get('CHAMELEON_VLM_MODEL','');projector=os.environ.get('CHAMELEON_VLM_MMPROJ','')
    return {'enabled':enabled,'backend':backend,'weights_present':bool(path and projector and Path(path).is_file() and Path(projector).is_file()),
            'runtime_available':_has('llama_cpp'),'plan_schema':os.environ.get('CHAMELEON_VLM_PLAN_SCHEMA','layout-plan/1')}


def describe(image,prompt,schema):
    global _MODEL
    if not status()['enabled'] or not status()['weights_present']:return None
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import MTMDChatHandler
        if _MODEL is None:
            handler=MTMDChatHandler(clip_model_path=os.environ['CHAMELEON_VLM_MMPROJ'],verbose=False)
            _MODEL=Llama(model_path=os.environ['CHAMELEON_VLM_MODEL'],chat_handler=handler,n_ctx=4096,n_threads=min(8,os.cpu_count() or 1),n_gpu_layers=0,seed=7,verbose=False)
        images=list(image) if isinstance(image,(tuple,list)) else [image];content=[{'type':'text','text':prompt}]
        for original in images:
            im=prepare_image(original);buf=io.BytesIO();im.save(buf,format='PNG');content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}})
        reply=_MODEL.create_chat_completion(messages=[{'role':'user','content':content}],temperature=0,max_tokens=700,response_format={'type':'json_object','schema':schema})
        answer=json.loads(reply['choices'][0]['message']['content'])
        import jsonschema
        jsonschema.validate(answer,schema);return answer
    except Exception:return None


# ------------------------------------------------------ transformers backend

def user_message(prompt,n_images=1):
    """The single chat turn used by training and inference (images first, then text)."""
    return {'role':'user','content':[{'type':'image'} for _ in range(n_images)]+[{'type':'text','text':prompt}]}


class TransformersPolicy:
    """Base model + LoRA adapter + saved processor from a checked bundle. Lazy ML imports."""

    def __init__(self,model,processor,adapter,manifest,device):
        self.model,self.processor,self.adapter,self.manifest,self.device=model,processor,adapter,manifest,device

    @classmethod
    def load(cls,bundle,device=None):
        check=check_bundle(bundle,verify_hashes=True)
        if check['status']!='ok':raise RuntimeError('bundle '+check['status']+': '+'; '.join(check['reasons']))
        os.environ.setdefault('HF_HUB_OFFLINE','1')
        import torch,transformers
        from peft import PeftModel
        m=check['manifest'];a=ADAPTERS[m['adapter_id']];base=m['base']
        device=device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        dtype=getattr(torch,m.get('dtype','float32'))
        processor=transformers.AutoProcessor.from_pretrained(str(Path(bundle)/'processor'),local_files_only=True)
        model_cls=getattr(transformers,a.model_class,None) or transformers.AutoModelForImageTextToText
        model=model_cls.from_pretrained(base.get('local_path') or base['repo'],revision=None if base.get('local_path') else base.get('revision'),dtype=dtype,local_files_only=True)
        model=PeftModel.from_pretrained(model,str(Path(bundle)/'adapter'),is_trainable=False).to(device).eval()
        return cls(model,processor,a,m,device)

    def generate(self,images,prompt,max_new_tokens=768):
        """Greedy decode; returns (text, stats). The caller parses and validates the text."""
        import torch
        images=[images] if not isinstance(images,(list,tuple)) else list(images)
        images=[image_transform(self.adapter.image_transform)(im) for im in images]
        text=self.processor.apply_chat_template([user_message(prompt,len(images))],tokenize=False,add_generation_prompt=True)
        batch=self.processor(text=[text],images=images,return_tensors='pt').to(self.device)
        started=time.monotonic()
        with torch.no_grad():
            out=self.model.generate(**batch,max_new_tokens=max_new_tokens,do_sample=False)
        new=out[0,batch['input_ids'].shape[1]:]
        reply=self.processor.tokenizer.decode(new,skip_special_tokens=True)
        return reply,{'new_tokens':int(new.shape[0]),'seconds':round(time.monotonic()-started,3)}


def generate_text(image,prompt,max_new_tokens=768):
    """Worker entry for the transformers backend. Returns raw text or None (typed failure upstream)."""
    global _POLICY
    st=status()
    if not st['enabled'] or not st['weights_present'] or not st['runtime_available']:return None
    if _POLICY is None:_POLICY=TransformersPolicy.load(os.environ['CHAMELEON_VLM_BUNDLE'])
    return _POLICY.generate(image,prompt,max_new_tokens)[0]
