"""Import a bounded, pinned Crello validation collection with real resources.

Run: .venv/bin/python -m tools.build_crello_catalog --download --limit 60
The half-GB shard is cached once. No model training uses this validation set.
"""
from __future__ import annotations
import argparse, hashlib, html, io, json, math, pickle, urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REVISION = '7997e2f434ee4aa73cf4cdf22c5954cb175872e1'
DATASET = 'cyberagent/crello'
FILES = {
 'fonts.pickle': ('resources/fonts.pickle', '990882f92440e67771a27c118c41d235f2c2605ca383a177366a773479773879'),
 'validation.parquet': ('data/validation-00000-of-00003.parquet', '08c19daf1ea367938c8c07c4415b1c21a7d74ba6e9d7ff217a5315e79a0c18e2'),
}
class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Executable objects are not supported in the font archive')

def download_and_verify(cache, download):
    cache.mkdir(parents=True, exist_ok=True)
    for name,(remote,digest) in FILES.items():
        p=cache/name
        if not p.exists():
            if not download: raise FileNotFoundError(f'{p}: rerun with --download')
            temp=p.with_suffix('.part')
            print(f'Downloading {remote}', flush=True)
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{remote}', temp)
            temp.replace(p)
        with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual != digest: raise ValueError(f'SHA256 mismatch: {p}')

def label(features, field, value):
    feature=features[field]
    names=feature.get('feature',feature).get('names')
    if names is not None and isinstance(value,int):
        if not 0<=value<len(names): raise ValueError(f'Invalid {field} category {value}')
        return names[value]
    return value

def convert_row(row, features, assets_dir):
    """Keep text editable; non-text layers use their original dataset PNGs."""
    key=hashlib.sha256(str(row['id']).encode()).hexdigest()[:20]
    elements=[];notes=[]
    for i in range(int(row['length'])):
        def at(k,default=None):
            values=row.get(k)
            return values[i] if values is not None and i<len(values) else default
        e={'id':f'el_{i}', 'x':float(at('left',0)), 'y':float(at('top',0)),
           'width':float(at('width',1)), 'height':float(at('height',1)),
           'rotation':float(at('angle',0)), 'opacity':float(at('opacity',1)), 'z_index':i}
        if not all(math.isfinite(e[k]) for k in ('x','y','width','height','rotation','opacity')) or min(e['width'],e['height'])<=0:
            raise ValueError(f'Invalid geometry at layer {i}')
        if label(features,'type',at('type'))=='TextElement':
            content=html.unescape(at('text','') or '').replace('\\n','\n').replace('\\r','\r')
            family=label(features,'font',at('font'))
            if not family: raise ValueError(f'Missing font at layer {i}')
            colors=at('text_color',[]) or [];bold=at('font_bold',[]) or [];italic=at('font_italic',[]) or []
            e.update(type='text',content=content,font_family=family,font_size=max(1,float(at('font_size',12))),
                     color=colors[0] if colors else 'rgba(0,0,0,1)',
                     text_align=label(features,'text_align',at('text_align')) or 'left',
                     line_height=float(at('line_height',1) or 1),letter_spacing=float(at('letter_spacing',0) or 0),capitalize=bool(at('capitalize',False)))
            runs=[]
            for j in range(len(content)):
                style={'color':colors[j] if j<len(colors) else e['color'],
                       'font_weight':'bold' if j<len(bold) and bold[j] else 'regular',
                       'font_style':'italic' if j<len(italic) and italic[j] else 'normal'}
                if runs and all(runs[-1].get(k)==v for k,v in style.items()):runs[-1]['end']=j
                else:runs.append({'start':j,'end':j,**style})
            e['style_runs']=runs
            if runs:e.update(font_weight=runs[0]['font_weight'],font_style=runs[0]['font_style'])
        else:
            blob=(at('image') or {}).get('bytes')
            if not blob: raise ValueError(f'Missing original image at layer {i}')
            asset=f'crello/{key}/{i}.png';p=assets_dir/asset;p.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(io.BytesIO(blob)) as im:
                im.verify()
            p.write_bytes(blob)
            e.update(type='image',asset_id=asset,fit='fill')
        elements.append(e)
    notes.append('Editable text is reflowed by the local renderer; wrapping, effects and text metrics may differ from the dataset preview.')
    if any(e['x']<0 or e['y']<0 or e['x']+e['width']>row['canvas_width']+1 or e['y']+e['height']>row['canvas_height']+1 for e in elements):
        notes.append('The original contains layers extending beyond the canvas; compare their crop when resizing.')
    if any(e.get('opacity',1)<1 and e['type']=='text' for e in elements):notes.append('Text transparency may differ in the renderer.')
    return key,{'canvas':{'width':int(row['canvas_width']),'height':int(row['canvas_height'])},'elements':elements},notes

def build(cache, out, limit):
    import pyarrow.parquet as pq
    from cr_renderer.fonts import normalize_family
    from renderer.converter.brand_fonts import build_font_dict
    assets=ROOT/'brand_kit/assets';fonts_dir=ROOT/'brand_kit/fonts'
    pf=pq.ParquetFile(cache/'validation.parquet')
    features=json.loads(pf.schema_arrow.metadata[b'huggingface'])['info']['features']
    with (cache/'fonts.pickle').open('rb') as f:font_dict=PrimitiveUnpickler(f).load()
    out.mkdir(parents=True,exist_ok=True);(out/'media').mkdir(exist_ok=True);(out/'designs').mkdir(exist_ok=True)
    rows=[]
    for batch in pf.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            if len(rows)>=limit:break
            rows.append(row)
        if len(rows)>=limit:break
    families={normalize_family(label(features,'font',r['font'][i])) for r in rows for i in range(r['length']) if label(features,'type',r['type'][i])=='TextElement'}
    manifest=[];(fonts_dir/'crello').mkdir(exist_ok=True)
    for family,faces in font_dict.items():
        if normalize_family(family) not in families:continue
        for face in faces:
            data=face['bytes'];name=hashlib.sha256(data).hexdigest()+'.ttf'
            (fonts_dir/'crello'/name).write_bytes(data)
            manifest.append({'file':'crello/'+name,**{k:face.get(k,'regular') for k in ('fontFamily','fontWeight','fontStyle')}})
    # Merge imports so increasing/reducing the catalog never breaks saved runs.
    mp=fonts_dir/'crello-fonts.json';old=json.loads(mp.read_text()) if mp.exists() else []
    merged={(x['file'],x['fontFamily'],x['fontWeight'],x['fontStyle']):x for x in old+manifest};mp.write_text(json.dumps(list(merged.values()),indent=2))
    available=build_font_dict(str(fonts_dir))
    from renderer.converter.convert import render_design_to_png
    entries=[];rejected=[]
    for row in rows:
        try:
            key,source,notes=convert_row(row,features,assets)
            source['provenance']={'dataset':DATASET,'revision':REVISION,'source_id':str(row['id']),'split':'validation'}
            used=sorted({e['font_family'] for e in source['elements'] if e['type']=='text'})
            missing=[]
            for e in source['elements']:
                if e['type']!='text':continue
                faces=available.get(normalize_family(e['font_family']),[])
                for run in e.get('style_runs') or [{}]:
                    weight=run.get('font_weight','regular');italic=run.get('font_style')=='italic'
                    style=('bolditalic' if italic else 'bold') if weight=='bold' else ('italic' if italic else 'regular')
                    if not any(f['fontWeight']==weight and f['fontStyle']==style for f in faces):
                        missing.append(f"{e['font_family']} ({style})")
            missing=sorted(set(missing))
            if missing:notes.append('Unavailable exact font faces: '+', '.join(missing))
            preview=(row.get('preview') or {}).get('bytes')
            if not preview:raise ValueError('Missing dataset preview')
            with Image.open(io.BytesIO(preview)) as im:
                im.convert('RGB').save(out/'media'/f'{key}-reference.jpg',quality=92)
            warnings=render_design_to_png(source,assets,out/'media'/f'{key}-editable.png',max_size=720)
            notes.extend(warnings)
            entry={'id':key,'source_id':str(row['id']),'title':row.get('title') or str(row['id']),
                   'format':label(features,'format',row['format']),'category':label(features,'category',row['category']),
                   'width':source['canvas']['width'],'height':source['canvas']['height'],'elements':len(source['elements']),
                   'text_elements':sum(e['type']=='text' for e in source['elements']),
                   'image_elements':sum(e['type']=='image' for e in source['elements']),
                   'fonts':used,'missing_fonts':missing,'assets_available':True,
                   'status':'needs-review' if missing or warnings else 'ready',
                   'reference':f'/api/crello/media/{key}-reference.jpg','reconstruction':f'/api/crello/media/{key}-editable.png',
                   'notes':notes,'split':'validation','keywords':row.get('keywords',[])}
            (out/'designs'/f'{key}.json').write_text(json.dumps({'source':source,'metadata':entry},ensure_ascii=False,indent=2))
            entries.append(entry);print(f"{len(entries)}/{limit} {entry['status']}: {entry['title']}",flush=True)
        except (ValueError,TypeError,KeyError,OSError) as exc:rejected.append({'id':str(row.get('id')),'reason':str(exc)})
    payload={'dataset':DATASET,'revision':REVISION,'split':'validation','license':'CDLA-Permissive-2.0',
             'dataset_url':f'https://huggingface.co/datasets/{DATASET}/tree/{REVISION}',
             'description':'A bounded validation collection for interactive testing, not training. Ready means resources are present; visual fidelity still needs review.',
             'entries':entries,'rejected':rejected}
    temp=out/'catalog.tmp';temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2));temp.replace(out/'catalog.json')
    print(f"Imported {len(entries)}, rejected {len(rejected)}. Restart the local service to load newly installed fonts.")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true');p.add_argument('--limit',type=int,default=60)
    p.add_argument('--cache',type=Path,default=ROOT/'work/crello-cache');p.add_argument('--output',type=Path,default=ROOT/'output/crello')
    a=p.parse_args()
    if not 1<=a.limit<=500: p.error('--limit must be between 1 and 500')
    download_and_verify(a.cache,a.download);build(a.cache,a.output,a.limit)
