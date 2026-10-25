"""Read-only, offline Crello gallery. Only manifest-listed local files are served."""
from __future__ import annotations
import json
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from engine.canonical import ROOT, ASSETS_DIR, FONTS_DIR

router=APIRouter(prefix='/api/crello',tags=['Crello'])
CATALOG_DIR=ROOT/'output/crello'

def catalog():
    path=CATALOG_DIR/'catalog.json'
    if not path.is_file():
        return {'dataset':'cyberagent/crello','revision':None,'split':'validation','entries':[],
                'description':'No local collection yet. Run the Crello catalog importer to download designs.'}
    return json.loads(path.read_text())

def entry(key):
    if not re.fullmatch(r'[a-f0-9]{20}',key):raise HTTPException(404,'Design not found')
    found=next((x for x in catalog()['entries'] if x['id']==key),None)
    if found is None:raise HTTPException(404,'Design not found')
    return found

@router.get('')
def list_designs(q:str=Query('',max_length=200),format:str='',orientation:str='',status:str='',offset:int=Query(0,ge=0),limit:int=Query(24,ge=1,le=100)):
    data=catalog();rows=data['entries'];formats=sorted({x['format'] for x in rows})
    def matches(x):
        ratio=x['width']/x['height'];o='square' if .95<=ratio<=1.05 else 'landscape' if ratio>1 else 'portrait'
        return (not q or q.casefold() in ' '.join([x['title'],x['source_id'],*x.get('keywords',[]),*x.get('fonts',[])]).casefold()) and (not format or x['format']==format) and (not orientation or orientation==o) and (not status or x['status']==status)
    filtered=[x for x in rows if matches(x)]
    return {**{k:v for k,v in data.items() if k not in ('entries','rejected')},'items':filtered[offset:offset+limit],
            'total':len(filtered),'collection_total':len(rows),'offset':offset,'limit':limit,'formats':formats}

@router.get('/media/{filename}')
def media(filename:str):
    match=re.fullmatch(r'([a-f0-9]{20})-(reference\.jpg|editable\.png)',filename)
    if not match:raise HTTPException(404,'Preview not found')
    entry(match[1]);path=CATALOG_DIR/'media'/filename
    if not path.is_file():raise HTTPException(404,'Preview unavailable')
    return FileResponse(path)

@router.get('/{key}')
def get_design(key:str):
    item=entry(key);path=CATALOG_DIR/'designs'/f'{key}.json'
    if not path.is_file():raise HTTPException(404,'Design unavailable')
    data=json.loads(path.read_text());missing_assets=[]
    for el in data['source']['elements']:
        if el['type']!='image':continue
        asset=(ASSETS_DIR/el['asset_id']).resolve()
        if not asset.is_relative_to(ASSETS_DIR.resolve()) or not asset.is_file():missing_assets.append(el['id'])
    # A removed resource must never silently become a placeholder on selection.
    from renderer.converter.brand_fonts import build_font_dict
    from cr_renderer.fonts import normalize_family
    available=build_font_dict(str(FONTS_DIR))
    missing_fonts=sorted({e['font_family'] for e in data['source']['elements'] if e['type']=='text' and normalize_family(e['font_family']) not in available})
    if missing_assets or missing_fonts:
        raise HTTPException(409,{'message':'Original resources are missing. Rebuild the Crello collection.', 'assets':missing_assets,'fonts':missing_fonts})
    return {**data,'metadata':item}
