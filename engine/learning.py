"""Portable learned roles and preference ranking. Missing weights are neutral."""
import json,os,math
from pathlib import Path
from .canonical import ROOT
ROLES=['headline','subhead','body','cta','badge','legal','logo','hero','image']
FEATURE_NAMES=['x','y','width','height','area','aspect','font','font_rank','z_rank','length','words','digits','is_text','is_image','is_shape','grouped']

def features(source):
    cw,ch=source['canvas']['width'],source['canvas']['height'];els=source['elements'];fonts=sorted({e.get('font_size',0) for e in els});zs=sorted({e.get('z_index',0) for e in els});out={}
    for e in els:
        t=str(e.get('content',''));fs=e.get('font_size',0)
        out[e['id']]=[e['x']/cw,e['y']/ch,e['width']/cw,e['height']/ch,e['width']*e['height']/(cw*ch),min(20,e['width']/max(.01,e['height']))/20,fs/min(cw,ch),fonts.index(fs)/max(1,len(fonts)-1),zs.index(e.get('z_index',0))/max(1,len(zs)-1),min(1,len(t)/300),min(1,len(t.split())/50),sum(c.isdigit() for c in t)/max(1,len(t)),float(e['type']=='text'),float(e['type']=='image'),float(e['type']=='shape'),float(bool(e.get('group_id')))]
    return out

def load_model(env,name):
    path=Path(os.environ.get(env,str(ROOT/'engine/models'/name)))
    if not path.is_file():return None
    try:return json.loads(path.read_text())
    except (ValueError,OSError):return None

def apply_roles(source,A,enabled=None):
    if enabled is False or (enabled is None and os.environ.get('CHAMELEON_ROLE_MODEL','0')!='1'):return {'status':'disabled'}
    model=load_model('CHAMELEON_ROLE_WEIGHTS','roles.json')
    if not model or model.get('features')!=FEATURE_NAMES:return {'status':'missing or incompatible weights'}
    xs=features(source);changes=[]
    for block in A.all_blocks():
        # Keep rigid/compound blocks under structural rules; learned votes affect loose elements.
        if len(block.ids)!=1 or block.role in ('background','decoration','sticker'):continue
        eid=block.ids[0];x=xs[eid];probs=[0.]*len(model['classes'])
        for tree in model['trees']:
            n=0
            while tree['left'][n]!=-1:n=tree['left'][n] if x[tree['feature'][n]]<=tree['threshold'][n] else tree['right'][n]
            values=tree['values'][n];den=sum(values) or 1
            probs=[p+v/den for p,v in zip(probs,values)]
        probs=[p/max(1,len(model['trees'])) for p in probs];k=max(range(len(probs)),key=probs.__getitem__);role=model['classes'][k];original=block.role
        allowed=('headline','subhead','body','legal') if block.members[0]['type']=='text' else ('image','hero','logo')
        if role in allowed and probs[k]>=.8 and (original not in model['classes'] or probs[k]-probs[model['classes'].index(original)]>=.3):
            block.role=role;A.roles[eid]=role;changes.append({'id':eid,'from':original,'to':role,'confidence':round(probs[k],3)})
    # Never invent extra primary headlines/heroes.
    for role in ('headline','hero','logo'):
        bs=[b for b in A.all_blocks() if b.role==role]
        for b in sorted(bs,key=lambda b:b.importance,reverse=True)[1:]:
            b.role='body' if b.members[0]['type']=='text' else 'image'
            for i in b.ids:A.roles[i]=b.role
    return {'status':'enabled','changes':changes}

def rank_features(result):
    q=result['score'];d=result['diagnostics'];p=q['parts']
    # Rubric dimensions (engine/rubric.py); neutral for labels recorded before it existed.
    rub=(result.get('rubric') or {}).get('tier2') or {};r=rub.get('scores',{});st=rub.get('stats',{})
    return [p[k]/100 for k in ('readability','hierarchy','composition','fidelity')]+[min(1,q['stats']['coverage']),1-min(1,len(d['overflow'])/3),1-min(1,len(d['collisions'])/3),1-min(1,len(d['contrast_regressions'])/3)]+[r.get(k,100)/100 for k in ('grouping','alignment','space','imagery')]+[1-min(1,len(rub.get('warnings',[]))/6),1-min(1,st.get('near_misses',0)/4)]
RANK_NAMES=['readability','hierarchy','composition','fidelity','coverage','fit','collision_free','contrast','grouping','alignment','space','imagery','few_warnings','no_near_misses']
# Features that may only ever raise preference: a ranker must not prefer more defects.
NONNEGATIVE=['readability','fit','collision_free','contrast','grouping']

def preference_score(result):
    m=load_model('CHAMELEON_RANK_WEIGHTS','ranker.json')
    if not compatible_ranker(m):return None
    try:x=rank_features(result)
    except (KeyError,TypeError,ValueError,ZeroDivisionError):return None
    if any(not math.isfinite(v) or abs((v-mu)/sd)>m.get('max_standardized_feature',8.) for v,mu,sd in zip(x,m['mean'],m['scale'])):return None
    return sum(w*(v-mu)/max(1e-8,sd) for w,v,mu,sd in zip(m['weights'],x,m['mean'],m['scale']))

def compatible_ranker(model):
    """Reject corrupt/development artifacts before they influence ordering."""
    if not isinstance(model,dict) or model.get('features')!=RANK_NAMES or model.get('version',1)!=1:return False
    if not isinstance(model.get('report',{}),dict) or model.get('report',{}).get('deployable') is False:return False
    for name in ('weights','mean','scale'):
        values=model.get(name)
        if not isinstance(values,list) or len(values)!=len(RANK_NAMES):return False
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):return False
    if any(v<=0 for v in model['scale']):return False
    bound=model.get('max_standardized_feature',8.)
    if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not math.isfinite(bound) or bound<=0:return False
    return all(model['weights'][RANK_NAMES.index(name)]>=0 for name in NONNEGATIVE)

def preference_comparison(before,after,min_margin=.1):
    """Abstain near zero or out-of-distribution; caller retains deterministic order."""
    a,b=preference_score(before),preference_score(after)
    if a is None or b is None:return {'available':False,'verdict':'unavailable','reason':'Missing, incompatible or out-of-distribution ranker'}
    margin=b-a
    return {'available':True,'verdict':'unsure' if abs(margin)<min_margin else 'after' if margin>0 else 'before','margin':margin,'calibrated':False}


# ---------------------------------------------------------------------------
# Calibrated role model v2 (CHAMELEON_ROLE_MODEL=calibrated|shadow)
#
# Versioned artifact engine/models/roles.v2.json carries its ontology, feature
# schema, model type, calibration temperature and class-specific override
# thresholds, each with hashes. Anything missing, incompatible or non-finite
# fails closed to the heuristic roles with an explicit status.
# ---------------------------------------------------------------------------
import hashlib

ROLE_ARTIFACT_VERSION='roles-calibrated/2'
FEATURE_NAMES_V2=FEATURE_NAMES+['font_ratio','text_rank','n_texts','cx','cy','bleeds','touch_sides','inside_count','holds_count','image_area_rank','line_estimate','upper_ratio','contact_cue','log_aspect']
TEXT_ROLES=('headline','subhead','body','legal')
IMAGE_ROLES=('image','hero','logo')

def _sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

FEATURE_SCHEMA_HASH=_sha(FEATURE_NAMES_V2)

def features_v2(source):
    """Geometry, typography and paint-order features; content only through
    generic cues (length, digits, capitals, URL/phone markers). Never IDs."""
    base=features(source);cw,ch=float(source['canvas']['width']),float(source['canvas']['height']);els=source['elements']
    texts=[e for e in els if e['type']=='text'];fmax=max([float(e.get('font_size') or 0) for e in texts] or [1]) or 1
    tsizes=sorted({float(e.get('font_size') or 0) for e in texts},reverse=True)
    imgs=sorted([e for e in els if e['type']!='text'],key=lambda e:-(e['width']*e['height']))
    z={e['id']:e.get('z_index',0) for e in els};out={}
    def box(e):return e['x'],e['y'],e['x']+e['width'],e['y']+e['height']
    for e in els:
        x0,y0,x1,y1=box(e);cx,cy=(x0+x1)/2,(y0+y1)/2;fs=float(e.get('font_size') or 0);t=str(e.get('content',''))
        inside=sum(1 for o in els if o is not e and z[o['id']]<=z[e['id']] and box(o)[0]<=cx<=box(o)[2] and box(o)[1]<=cy<=box(o)[3])
        holds=sum(1 for o in els if o is not e and z[o['id']]>=z[e['id']] and x0<=(box(o)[0]+box(o)[2])/2<=x1 and y0<=(box(o)[1]+box(o)[3])/2<=y1)
        letters=[c for c in t if c.isalpha()]
        v=base[e['id']]+[
            fs/fmax if e['type']=='text' else 0.,
            (tsizes.index(fs)/max(1,len(tsizes)-1)) if e['type']=='text' and fs in tsizes else 1.,
            min(1.,len(texts)/20),
            cx/cw,cy/ch,
            float(x0<-1 or y0<-1 or x1>cw+1 or y1>ch+1),
            sum((x0<=1,y0<=1,x1>=cw-1,y1>=ch-1))/4,
            min(1.,inside/5),min(1.,holds/10),
            (imgs.index(e)/max(1,len(imgs)-1)) if e in imgs else 1.,
            min(1.,(e['height']/max(1e-6,fs*1.2))/10) if e['type']=='text' and fs>0 else 0.,
            sum(c.isupper() for c in letters)/max(1,len(letters)),
            float(any(k in t.lower() for k in ('www','http','.com','@'))),
            max(-3.,min(3.,math.log(max(.01,e['width'])/max(.01,e['height']))))/3,
        ]
        out[e['id']]=v
    return out

def _finite(x):
    if isinstance(x,(int,float)):return math.isfinite(x)
    if isinstance(x,list):return all(_finite(v) for v in x)
    if isinstance(x,dict):return all(_finite(v) for v in x.values())
    return True

def load_role_artifact(path=None):
    """(artifact, status). Fails closed: status != 'ok' means do not use."""
    p=Path(path or os.environ.get('CHAMELEON_ROLE_WEIGHTS_V2',str(ROOT/'engine/models/roles.v2.json')))
    if not p.is_file():return None,'missing artifact'
    try:
        if p.stat().st_size>50_000_000:return None,'artifact exceeds local resource limit'
        m=json.loads(p.read_text())
    except (ValueError,OSError):return None,'unreadable artifact'
    if not isinstance(m,dict) or m.get('version')!=ROLE_ARTIFACT_VERSION:return None,'incompatible version'
    if m.get('features')!=FEATURE_NAMES_V2 or m.get('feature_schema_hash')!=FEATURE_SCHEMA_HASH:return None,'incompatible feature schema'
    if m.get('ontology_hash')!=_sha(m.get('ontology')) or m.get('calibration_hash')!=_sha(m.get('calibration')):return None,'hash mismatch'
    if m.get('artifact_hash')!=_sha({k:v for k,v in m.items() if k not in ('training','artifact_hash')})[:16]:return None,'artifact hash mismatch'
    cal=m.get('calibration')
    if not isinstance(cal,dict):return None,'invalid calibration'
    T=cal.get('temperature')
    if isinstance(T,bool) or not isinstance(T,(int,float)) or not math.isfinite(T) or T<=0:return None,'invalid calibration'
    if m.get('model_type') not in ('rf','gb') or not m.get('trees') or not _finite(m['trees']) or not _finite(cal):return None,'non-finite or empty model'
    classes=m.get('classes');thresholds=cal.get('thresholds');ontology=m.get('ontology')
    if not isinstance(classes,list) or len(classes)<2 or any(not isinstance(c,str) for c in classes) or len(set(classes))!=len(classes):return None,'invalid classes'
    if not isinstance(ontology,dict) or any(c not in ontology for c in classes):return None,'invalid ontology'
    if not isinstance(thresholds,dict) or any(c not in classes or isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=1 for c,v in thresholds.items()):return None,'invalid thresholds'
    def tree_ok(tree,width,counts):
        if not isinstance(tree,dict) or any(not isinstance(tree.get(k),list) for k in ('left','right','feature','threshold','values')):return False
        n=len(tree['left'])
        if not 1<=n<=200000 or any(len(tree[k])!=n for k in ('right','feature','threshold','values')):return False
        for i,(left,right,feature,threshold,values) in enumerate(zip(*(tree[k] for k in ('left','right','feature','threshold','values')))):
            if any(isinstance(v,bool) or not isinstance(v,int) for v in (left,right,feature)):return False
            if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold):return False
            if not isinstance(values,list) or len(values)!=width or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or (counts and v<0) for v in values):return False
            if left==-1:
                if right!=-1 or (counts and sum(values)<=0):return False
            # Exported sklearn nodes are preorder-indexed. Forward children
            # guarantee bounded traversal, excluding cycles and invalid offsets.
            elif not (i<left<n and i<right<n and 0<=feature<len(FEATURE_NAMES_V2)):return False
        return True
    trees=m['trees']
    if not isinstance(trees,list) or len(trees)>512:return None,'invalid forest'
    if m['model_type']=='rf':
        if not all(tree_ok(tree,len(classes),True) for tree in trees):return None,'invalid tree structure'
    else:
        initial=m.get('init');rate=m.get('learning_rate')
        if not isinstance(initial,list) or len(initial)!=len(classes) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in initial) or isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0:return None,'invalid boosting metadata'
        if not all(isinstance(stage,list) and len(stage)==len(classes) and all(tree_ok(tree,1,False) for tree in stage) for stage in trees):return None,'invalid boosting structure'
    return m,'ok'

def _leaf(tree,x):
    n=0
    while tree['left'][n]!=-1:n=tree['left'][n] if x[tree['feature'][n]]<=tree['threshold'][n] else tree['right'][n]
    return tree['values'][n]

def predict_proba(m,x):
    """Calibrated class probabilities for one feature vector."""
    import struct
    # Trees were fit on float32-cast features; compare the same way.
    x=[struct.unpack('f',struct.pack('f',float(v)))[0] for v in x]
    k=len(m['classes'])
    if m['model_type']=='rf':
        acc=[0.]*k
        for t in m['trees']:
            v=_leaf(t,x);d=sum(v) or 1;acc=[a+b/d for a,b in zip(acc,v)]
        logits=[math.log(max(1e-9,a/len(m['trees']))) for a in acc]
    else:
        logits=list(m['init'])
        for stage in m['trees']:
            for c,t in enumerate(stage):logits[c]+=m['learning_rate']*_leaf(t,x)[0]
    T=m['calibration']['temperature'];z=[l/T for l in logits];mx=max(z);e=[math.exp(v-mx) for v in z];s=sum(e)
    return [v/s for v in e]

def role_mode(enabled=None):
    if isinstance(enabled,str):
        mode={'0':'legacy','legacy':'legacy','1':'rf','rf':'rf','calibrated':'calibrated','shadow':'shadow'}.get(enabled.strip().lower())
        if mode is None:raise ValueError('Unknown explicit role mode')
        return mode
    if enabled is False:return 'legacy'
    if enabled is True:return 'rf'
    v=os.environ.get('CHAMELEON_ROLE_MODEL','0').strip().lower()
    return {'1':'rf','rf':'rf','calibrated':'calibrated','shadow':'shadow'}.get(v,'legacy')

_apply_roles_v1=apply_roles

def apply_roles(source,A,enabled=None):
    """Role merge. legacy (default) | rf (CHAMELEON_ROLE_MODEL=1, shipped v1 trees)
    | calibrated (v2 overrides above class thresholds) | shadow (v2 suggestions
    recorded only). Explicit source groups and compound blocks are never
    overridden; impossible type/role pairs are rejected."""
    mode=role_mode(enabled)
    if hasattr(A,'_graph_cache'):A._graph_cache=None
    if mode in ('legacy','rf'):
        status=_apply_roles_v1(source,A,enabled=True if mode=='rf' else False)
        overrides=getattr(A,'role_overrides',{})
        if overrides:
            from .analysis import apply_role_overrides
            apply_role_overrides(A,overrides)
            status['changes']=[change for change in status.get('changes',[]) if change['id'] not in overrides]
            status['protected_overrides']=sorted(overrides)
        if hasattr(A,'role_status'):A.role_status=status
        return status
    m,st=load_role_artifact()
    if m is None:
        status={'status':'fallback_heuristic','reason':st,'mode':mode}
        if hasattr(A,'role_status'):A.role_status=status
        return status
    from .analysis import IMPORTANCE,LABELS
    xs=features_v2(source);suggestions=[];changes=[];conf={}
    thresholds=m['calibration']['thresholds']
    for block in A.all_blocks():
        if len(block.ids)!=1 or block.role in ('background','decoration','sticker'):continue
        eid=block.ids[0];el=block.members[0]
        if eid in getattr(A,'role_overrides',{}):conf[eid]=1.0;continue
        p=predict_proba(m,xs[eid]);classes=m['classes']
        k=max(range(len(p)),key=p.__getitem__);role=classes[k];orig=block.role
        if orig in classes:conf[eid]=round(p[classes.index(orig)],4)
        allowed=TEXT_ROLES if el['type']=='text' else IMAGE_ROLES
        if el.get('group_id') or eid in getattr(A,'role_overrides',{}) or role==orig or role not in allowed or orig not in allowed:continue
        thr=thresholds.get(role)
        if thr is None or p[k]<thr:continue
        suggestions.append({'id':eid,'from':orig,'to':role,'confidence':round(p[k],4),'threshold':thr})
    if mode=='calibrated':
        for s in suggestions:
            b=next(b for b in A.all_blocks() if b.ids==[s['id']])
            b.role,b.label,b.importance=s['to'],LABELS.get(s['to'],b.label),IMPORTANCE.get(s['to'],b.importance)
            A.roles[s['id']]=s['to'];changes.append(s)
            if hasattr(A,'role_sources'):A.role_sources[s['id']]='model'
            conf[s['id']]=s['confidence']
        # Never invent extra primary headlines/heroes/logos.
        for role in ('headline','hero','logo'):
            bs=[b for b in A.all_blocks() if b.role==role]
            protected=[b for b in bs if any(getattr(A,'role_overrides',{}).get(i)==role for i in b.ids)]
            ordered=sorted(bs,key=lambda b:({'explicit':2,'heuristic':1,'model':0}.get(getattr(A,'role_sources',{}).get(b.ids[0],'heuristic'),1),b.src_box.area),reverse=True)
            keep=protected or ordered[:1]
            for b in (b for b in ordered if b not in keep and len(b.ids)==1 and getattr(A,'role_sources',{}).get(b.ids[0])=='model'):
                nr='body' if b.members[0]['type']=='text' else 'image'
                b.role,b.label,b.importance=nr,LABELS[nr],IMPORTANCE[nr]
                for i in b.ids:A.roles[i]=nr
    if hasattr(A,'role_confidence'):A.role_confidence.update(conf)
    status={'status':'enabled' if mode=='calibrated' else 'shadow','mode':mode,'artifact':m.get('artifact_hash'),'changes':changes,'suggestions':suggestions}
    if hasattr(A,'role_status'):A.role_status=status
    return status
