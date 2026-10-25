"""Build a reproducible weak-label role pilot from public metadata, without image downloads.

DesignAsCode themes are machine-generated labels, not human ground truth.
Read only Crello training-shard metadata columns using range requests.
"""
import argparse,collections,hashlib,json,re,urllib.request
from pathlib import Path

def tagged(text,name):
    match=re.search(r'<'+name+r'>\s*(.*?)\s*</'+name+r'>',text,re.S)
    return json.loads(match.group(1)) if match else []

def label(theme,kind):
    t=theme.lower()
    if any(x in t for x in ('call-to-action','call to action','button','cta')):return 'cta'
    if kind=='text':
        if any(x in t for x in ('subtitle','subheading','tagline','subhead')):return 'subhead'
        if any(x in t for x in ('copyright','disclaimer','legal','footer','website','contact','address','phone')):return 'legal'
        if any(x in t for x in ('headline','main title','heading','title text','main text','main message','slogan')):return 'headline'
        if any(x in t for x in ('badge','sticker','discount','price','date','time','offer')):return 'badge'
        if any(x in t for x in ('body','paragraph','description','supporting text')):return 'body'
    else:
        if 'background' in t:return 'background'
        if 'logo' in t:return 'logo'
        if any(x in t for x in ('hero','main image','product image','main photo')):return 'hero'
        if any(x in t for x in ('decorative','ornament','accent')):return 'decoration'
        if any(x in t for x in ('photo','image','illustration')):return 'image'
    return None

def build(shards,out):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    root=out.parent;root.mkdir(parents=True,exist_ok=True)
    revs={}
    for repo in ('cyberagent/crello','Tony1109/DesignAsCode-training-data'):
        revs[repo]=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=30))['sha']
    fs=HfFileSystem(block_size=131072);rows={}
    for i in range(shards):
        path=f'datasets/cyberagent/crello@{revs["cyberagent/crello"]}/data/train-{i:05}-of-00031.parquet'
        print('Reading training metadata shard',i,flush=True)
        with fs.open(path,'rb',block_size=131072) as f:
            pf=pq.ParquetFile(f);wanted=[k for k in ('id','canvas_width','canvas_height','length','left','top','width','height','angle','text','font_size','opacity') if k in pf.schema_arrow.names]
            for row in pf.read(columns=wanted).to_pylist():rows[row['id']]=row
    labels=root/'designascode.jsonl'
    if not labels.exists():
        print('Downloading public weak-label descriptions (151 MB; no images)',flush=True)
        urllib.request.urlretrieve('https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data/resolve/'+revs['Tony1109/DesignAsCode-training-data']+'/dataset.jsonl',labels)
    result=[];counts=collections.Counter()
    with labels.open() as f:
        for line in f:
            entry=json.loads(line);row=rows.get(entry['id'])
            if row is None:continue
            els=[]
            for i in range(row['length']):
                text=row['text'][i] or '';e={'id':f'el_{i}','type':'text' if text else 'image','content':text,'font_family':'metadata-only','font_size':max(1,float(row['font_size'][i] or 1)),'x':float(row['left'][i]),'y':float(row['top'][i]),'width':max(.01,float(row['width'][i])),'height':max(.01,float(row['height'][i])),'z_index':i}
                els.append(e)
            roles={}
            for group in tagged(entry['grouping'],'grouping'):
                for i in group['children']:
                    if not isinstance(i,int) or not 0<=i<len(els):continue
                    role=label(group.get('theme',''),els[i]['type'])
                    if role:roles[els[i]['id']]=role;counts[role]+=1
            if not roles:continue
            # Normalized copy groups exact content variants across canvas sizes.
            textkey='|'.join(sorted(re.sub(r'\s+',' ',e['content']).strip().lower() for e in els if e['content']))
            family=hashlib.sha256((textkey or entry['id']).encode()).hexdigest()[:20]
            result.append({'source':{'canvas':{'width':row['canvas_width'],'height':row['canvas_height']},'elements':els},'roles':roles,'family':family,'kind':'weak','source_id':entry['id']})
    out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in result)+'\n')
    manifest={'datasets':revs,'training_shards':list(range(shards)),'designs':len(result),'labels':dict(counts),'label_kind':'machine-distilled themes mapped by keyword; not human labels','units':'Crello v5 pixels; font family unused by classifier','split':'All source data from Crello train; train_roles holds out exact-copy families','source_hash':hashlib.sha256(labels.read_bytes()).hexdigest()}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shards',type=int,default=2);p.add_argument('--output',type=Path,default=Path('work/role-data/labels.jsonl'));a=p.parse_args();build(a.shards,a.output)
