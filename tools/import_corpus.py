"""Curate challenge-compatible JSON tasks with family splits, hashes, and manifests.

No synthetic target is labeled ground truth. Unsupported source formats are rejected.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from engine.invariants import validate,passes
from engine.analysis import analyze

def import_corpus(directory,out):
    out.mkdir(parents=True,exist_ok=True);manifest=[];seen=set()
    for path in sorted(directory.glob('*.json')):
        try:
            task=json.loads(path.read_text());d=task.get('source',task)
            if not isinstance(d.get('elements'),list) or not d['elements']:raise ValueError('No structured elements')
            if not passes(validate(d,d)):raise ValueError('Source does not satisfy structural contract; curate before import')
            family=str(task.get('family',path.stem));h=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
            if h in seen:continue
            seen.add(h);bucket=int(hashlib.sha256(family.encode()).hexdigest()[:8],16)%100
            split='train' if bucket<70 else 'validation' if bucket<85 else 'test'
            record={'source':d,'family':family,'split':split,'source_hash':h,'provenance':task.get('provenance',str(path)),'license':task.get('license','unspecified'),'adapter':'challenge-json-v1','annotation_status':'unreviewed'}
            (out/(h[:16]+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2));manifest.append({k:v for k,v in record.items() if k!='source'})
        except (ValueError,KeyError,TypeError) as e:manifest.append({'path':str(path),'rejected':str(e)})
    (out/'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in manifest));return manifest
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();print(len(import_corpus(a.input,a.output)),'manifest records')
