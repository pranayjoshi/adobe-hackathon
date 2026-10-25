"""Live integration check: four Crello designs, three output sizes each.
Run from the repository root with the local service listening on port 8765.
"""
import json,urllib.request,time
from pathlib import Path
base='http://127.0.0.1:8765'
def get(path):return json.load(urllib.request.urlopen(base+path))
def post(path,data):
 req=urllib.request.Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=300))
catalog=get('/api/crello?limit=100');print(catalog['total'],'designs',flush=True)
report=[]
# Different text/image mixes and shapes, selected deterministically.
for index in [1,10,25,40]:
 item=catalog['items'][index];d=get('/api/crello/'+item['id'])['source']
 for e in d['elements']:
  if e['type']=='image':assert (Path('brand_kit/assets')/e['asset_id']).is_file()
 for target in [{'width':1080,'height':1080},{'width':1280,'height':720},{'width':1080,'height':1920}]:
  started=time.time();r=post('/api/generate',{'source':d,'targets':[target],'top_k':3,'options':{'solver':True,'retrieval':True}})
  candidates=r['results'][0]['candidates'];assert candidates
  for candidate in candidates:
   out=candidate['design'];assert {e['id'] for e in out['elements']}=={e['id'] for e in d['elements']}
   by_id={e['id']:e for e in out['elements']}
   for e in d['elements']:
    for field in ['content','font_family','asset_id']:
     assert e.get(field)==by_id[e['id']].get(field),(field,item['title'])
   assert candidate['render'] and urllib.request.urlopen(base+candidate['render']).status==200
  row={'id':item['id'],'title':item['title'],'target':target,'valid':sum(c['valid'] for c in candidates),'candidates':len(candidates),'seconds':round(time.time()-started,2),'checks':candidates[0]['checks'],'quality':[{'overflow':c['diagnostics']['overflow'],'collisions':c['diagnostics']['collisions'],'contrast_regressions':c['diagnostics']['contrast_regressions']} for c in candidates]}
  report.append(row);print(json.dumps(row),flush=True)
p=Path('output/evaluation/crello-browser');p.mkdir(parents=True,exist_ok=True);(p/'results.json').write_text(json.dumps(report,indent=2))
print('COMPLETE',len(report),'runs',flush=True)
