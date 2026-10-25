"""Explicit Crello pixel/degree adapter for a pinned local Parquet shard.

Reads metadata only. Missing assets remain missing; no artwork is replaced.
Requires a font mapping for categorical fonts and marks reconstruction unverified.
"""
import argparse,json,hashlib
from pathlib import Path

def convert(row,font_names):
    cw,ch=int(row['canvas_width']),int(row['canvas_height']);els=[];count=int(row['length'])
    for i in range(count):
        def at(key,default=None):
            v=row.get(key);return v[i] if v is not None and i<len(v) else default
        txt=at('text','') or '';font=at('font');family=font_names.get(str(font),font if isinstance(font,str) else None)
        e={'id':f'el_{i}','x':float(at('left',0)),'y':float(at('top',0)),'width':float(at('width',1)),'height':float(at('height',1)),'rotation':float(at('angle',0)),'opacity':float(at('opacity',1)),'z_index':i}
        if txt:
            if not family:raise ValueError('Categorical font has no version-specific mapping')
            e.update(type='text',content=txt,font_family=family,font_size=max(1,float(at('font_size',12))),color='#000000',text_align='left',line_height=float(at('line_height',1) or 1),letter_spacing=float(at('letter_spacing',0) or 0))
            colors=at('text_color',[]) or []
            if colors and isinstance(colors[0],str):e['color']=colors[0]
            # Character styling retained as runs; text itself is never flattened.
            bold=at('font_bold',[]) or [];italic=at('font_italic',[]) or []
            runs=[]
            for j in range(len(txt)):
                style={'font_weight':'bold' if j<len(bold) and bold[j] else 'regular','font_style':'italic' if j<len(italic) and italic[j] else 'normal'}
                if j<len(colors) and isinstance(colors[j],str):style['color']=colors[j]
                if runs and {k:v for k,v in runs[-1].items() if k not in ('start','end')}==style:runs[-1]['end']=j+1
                else:runs.append({'start':j,'end':j+1,**style})
            e['style_runs']=runs
        else:e.update(type='image',asset_id=f"crello_{row['id']}_{i}.png",fit='contain')
        els.append(e)
    return {'canvas':{'width':cw,'height':ch},'elements':els}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('parquet',type=Path);ap.add_argument('--font-map',type=Path,required=True);ap.add_argument('--revision',required=True);ap.add_argument('--split',choices=['train','validation','test'],required=True);ap.add_argument('--limit',type=int,default=100);ap.add_argument('--output',type=Path,default=Path('output/crello-metadata'));a=ap.parse_args()
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(a.parquet);cols=[n for n in pf.schema_arrow.names if n not in ('image','preview')];mapping=json.loads(a.font_map.read_text());a.output.mkdir(parents=True,exist_ok=True);manifest=[];count=0
    for batch in pf.iter_batches(batch_size=32,columns=cols):
        for row in batch.to_pylist():
            if count>=a.limit:break
            try:
                d=convert(row,mapping);key=str(row['id']);safe=hashlib.sha256(key.encode()).hexdigest()[:16]
                task={'source':d,'family':key,'split':a.split,'provenance':{'dataset':'cyberagent/crello','revision':a.revision},'reconstruction':'metadata-only; image pixels and font files must be supplied and fidelity reviewed','license':'Consult pinned dataset and underlying artwork terms'}
                (a.output/(safe+'.json')).write_text(json.dumps(task,ensure_ascii=False));manifest.append({'id':key,'file':safe+'.json','status':'unverified reconstruction'});count+=1
            except (ValueError,TypeError) as e:manifest.append({'id':row.get('id'),'rejected':str(e)})
        if count>=a.limit:break
    (a.output/'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in manifest));print(count,'metadata designs, not fidelity-approved training examples')
