"""Typed OpenJev decisions with exact provenance and explicit unavailability.

Readout probabilities are model distributions, never calibrated correctness.
No HTTP or rendering happens at import or construction time.
"""
from __future__ import annotations
import base64, hashlib, io, json, math, os, time
from typing import Any, TypedDict
from PIL import Image
from ..constitution import PAIR_PROMPT
from ..formats import name_for

class PairVerdict(TypedDict, total=False):
    available: bool
    verdict: str
    before_hash: str
    after_hash: str
    evaluator_version: str
    provenance: dict[str,Any]
    reason: str

def design_hash(design):
    return hashlib.sha256(json.dumps(design,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def request_readout(state, images, questions, *, url=None, model=None, timeout=30, max_size=1024):
    """Small shared transport; raises on unavailable or malformed server output."""
    import httpx
    urls=[]
    for original in images:
        im=original.copy();im.thumbnail((max_size,max_size));buf=io.BytesIO();im.save(buf,format='PNG')
        urls.append('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
    key=os.environ.get('OPENJEV_API_KEY')
    response=httpx.post((url or os.environ.get('CHAMELEON_OPENJEV_URL','http://127.0.0.1:8000')).rstrip('/')+'/v1/systemone',
        timeout=max(.01,float(timeout)),headers={'Authorization':'Bearer '+key} if key else {},
        json={'model':model or os.environ.get('CHAMELEON_OPENJEV_MODEL','jev-latest'),'state':state,'images':urls,'questions':questions})
    response.raise_for_status();answers=response.json()['answers']
    if not isinstance(answers,dict):raise ValueError('Invalid OpenJev answers')
    return answers

class OpenJevEvaluator:
    version='openjev-pair-v1'
    def __init__(self,url=None,model=None,timeout=30,swap=True,transport=None):
        self.url=url or os.environ.get('CHAMELEON_OPENJEV_URL','http://127.0.0.1:8000')
        self.model=model or os.environ.get('CHAMELEON_OPENJEV_MODEL','jev-latest')
        self.timeout=float(timeout);self.swap=swap;self.transport=transport or request_readout

    def compare(self,source,target,before_design,after_design,*,images=None,timeout=None)->PairVerdict:
        started=time.monotonic();deadline=started+min(self.timeout,float(timeout) if timeout is not None else self.timeout)
        result=PairVerdict(available=False,verdict='unavailable',before_hash=design_hash(before_design),after_hash=design_hash(after_design),evaluator_version=self.version,
            provenance={'origin':'model','backend':'openjev','model':self.model,'source_hash':design_hash(source),'target':target,'presentation_order':[],
                        'probabilities_calibrated':False,'raw_readouts':[],'model_calls':0,'renders':0})
        try:
            if images is None:
                from ..quality import verify_design
                images=[]
                for design in (source,before_design,after_design):
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('Pair evaluation render budget exhausted')
                    verification=verify_design(design,timeout=remaining)
                    result['provenance']['renders']+=int(not verification.get('cache_hit'))
                    if not verification['verified']:raise RuntimeError(verification.get('error','Required preview unavailable'))
                    images.append(verification['image'])
            if len(images)!=3 or not all(isinstance(im,Image.Image) for im in images):raise ValueError('Expected source, before and after images')
            prompt=PAIR_PROMPT.format(src_w=source['canvas']['width'],src_h=source['canvas']['height'],tgt_w=target['width'],tgt_h=target['height'],format_name=name_for(target['width'],target['height']))
            questions={'verdict':{'type':'choice','instructions':'Which resize would a professional designer ship? Use both_bad when neither is acceptable.',
                       'criteria':{'A':'Resize A (Image 2) is better','B':'Resize B (Image 3) is better','tie':'Equivalent','both_bad':'Neither is acceptable','unsure':'Cannot tell'}}}
            votes=[]
            orders=[('before','after'),('after','before')] if self.swap else [('before','after')]
            for order in orders:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Pair evaluation budget exhausted')
                ordered=[images[0],images[1],images[2]] if order[0]=='before' else [images[0],images[2],images[1]]
                result['provenance']['model_calls']+=1
                answers=self.transport(prompt,ordered,questions,url=self.url,model=self.model,timeout=remaining)
                readout=answers['verdict'];choice=readout['choice']
                if choice not in ('A','B','tie','both_bad','unsure'):raise ValueError('Unknown pair choice')
                distribution=readout.get('probabilities',{})
                if not isinstance(distribution,dict) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in distribution.values()):raise ValueError('Invalid model distribution')
                votes.append(order[0] if choice=='A' else order[1] if choice=='B' else choice)
                result['provenance']['presentation_order'].append(list(order));result['provenance']['raw_readouts'].append(readout)
            verdict=votes[0] if all(v==votes[0] for v in votes) else 'unsure'
            result.update(available=True,verdict=verdict)
            if len(set(votes))>1:result['reason']='Position-swapped judgments disagree'
        except Exception as exc:
            result['reason']=type(exc).__name__+': '+str(exc)
        result['provenance']['duration_ms']=round((time.monotonic()-started)*1000,3)
        return result
