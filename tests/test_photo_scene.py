import copy,json,unittest
from pathlib import Path
from engine.analysis import analyze
from engine.scene import scene_defects,background_cover_allowed,propose_photo_scene
from engine.invariants import validate,passes
from engine.pipeline import generate
ROOT=Path(__file__).resolve().parents[1]
class PhotoSceneTests(unittest.TestCase):
 def source(self):
  p=ROOT/'output/crello/designs/72633b24108dde39d197.json'
  if not p.exists():self.skipTest('Local Crello collection not installed')
  return json.loads(p.read_text())['source']
 def test_only_canvas_background_may_change_to_cover(self):
  s={'canvas':{'width':100,'height':100},'elements':[{'id':'bg','type':'image','asset_id':'photo.png','fit':'fill','x':0,'y':0,'width':100,'height':100,'z_index':0},{'id':'hero','type':'image','asset_id':'photo.png','fit':'contain','x':20,'y':20,'width':40,'height':40,'z_index':1}]}
  o=copy.deepcopy(s);o['canvas']={'width':400,'height':100};o['elements'][0].update(width=400,fit='cover')
  self.assertTrue(passes(validate(s,o)))
  o['elements'][1]['fit']='cover';self.assertFalse(passes(validate(s,o)))
 def test_white_text_escaping_background_is_detected(self):
  s=self.source();o=copy.deepcopy(s);o['canvas']={'width':1600,'height':400}
  o['elements'][0].update(x=500,y=0,width=533,height=400)
  next(e for e in o['elements'] if e['type']=='text').update(x=1100,y=100,width=400,height=100)
  self.assertTrue(scene_defects(s,o)['unsupported_text'])
 def test_photo_scene_fills_banner_and_preserves_resources(self):
  s=self.source();before=copy.deepcopy(s);d=propose_photo_scene(s,{'width':1600,'height':400},analyze(s))[0]
  self.assertEqual(s,before);self.assertTrue(passes(validate(s,d)))
  bg=d['elements'][0];self.assertEqual(bg['fit'],'cover');self.assertEqual((bg['width'],bg['height']),(1600,400))
  self.assertFalse(scene_defects(s,d)['unsupported_text'])
  for old,new in zip(s['elements'],d['elements']):
   for k in ('id','asset_id','content','font_family'):self.assertEqual(old.get(k),new.get(k))
 def test_old_bad_banner_is_not_the_winner(self):
  s=self.source();r=generate(s,{'width':1600,'height':400},options={'render':False,'solver':False},all_candidates=True)
  self.assertTrue(r[0]['quality']['passed']);self.assertEqual(r[0]['template'].replace('refined_',''),'photo_scene')
  rejected=[c for c in r if c['template'].startswith('banner_')];self.assertTrue(rejected);self.assertTrue(any(c['quality']['defects']['unsupported_text'] for c in rejected))


class SubjectAwarePlacementTests(unittest.TestCase):
 """Guide 08: renderer-exact mask transforms, quiet regions, typed fallbacks."""
 def subject_image(self,busy=False):
  from PIL import Image,ImageDraw
  im=Image.new('RGBA',(200,100),(20,20,30,255));d=ImageDraw.Draw(im)
  if busy:
   for i in range(0,200,4):d.line((i,0,i,100),fill=(240,240,240,255) if i%8 else (10,10,10,255))
  d.ellipse((130,20,190,90),fill=(240,170,120,255))   # bright warm subject on the right
  return im
 def maps(self,im,mode='subject_aware'):
  from engine.photo_analysis import analyze_image
  return analyze_image(im,'synthetic',mode=mode,size=100)
 def test_subject_mask_follows_contain_and_cover_like_the_renderer(self):
  from engine.photo_analysis import canvas_field,quiet_regions,color_luminance
  from engine.canonical import Box
  m=self.maps(self.subject_image())
  x0,y0,x1,y1=m.subject_boxes[0];self.assertGreater(x0,.55)          # subject found on the right
  # contain: 200x100 asset in a 400x400 box -> 400x200 picture centred vertically (rows 100..300)
  f=canvas_field({'id':'p','x':0,'y':0,'width':400,'height':400,'fit':'contain'},m,400,400)
  ys=[r for r in range(f.gh) if f.subject[r].any()]
  self.assertTrue(ys and min(ys)*400/f.gh>=100-10 and max(ys)*400/f.gh<=300+10)
  self.assertEqual(f.coverage[0].max(),0)                            # letterbox band: no photo
  # cover: same asset in 100x100 -> scaled to 200x100, centre crop keeps asset x 50..150
  f=canvas_field({'id':'p','x':0,'y':0,'width':100,'height':100,'fit':'cover'},m,100,100)
  cols=[c for c in range(f.gw) if f.subject[:,c].any()]
  self.assertTrue(cols and min(cols)*100/f.gw>=(130-50)-6)          # subject starts at canvas x ~80
  regions=quiet_regions(f,30,20,text_lum=color_luminance('#ffffff'))
  self.assertTrue(regions)
  self.assertTrue(all(q.subject_overlap<=.06 and q.box.x1<=80+6 for q in regions))
 def test_no_quiet_region_is_an_explicit_empty_result(self):
  from engine.photo_analysis import canvas_field,quiet_regions
  im=self.subject_image();before=im.tobytes()
  m=self.maps(im)
  f=canvas_field({'id':'p','x':0,'y':0,'width':200,'height':100,'fit':'fill'},m,200,100)
  self.assertEqual(quiet_regions(f,190,95),[])                      # any big box hits the subject
  self.assertEqual(im.tobytes(),before)                             # original pixels untouched
 def test_missing_or_uncertain_model_uses_heuristic_and_adds_no_fields(self):
  from engine import photo_analysis as P
  im=self.subject_image()
  self.assertEqual(self.maps(im).status,'fallback:model_unavailable')
  self.assertEqual(self.maps(im,'heuristic').status,'ok')
  try:
   P.register_subject_model(lambda image:(None,.2),'stub')
   self.assertTrue(self.maps(im).status.startswith('fallback:model_error'))
   import numpy as np
   P.register_subject_model(lambda image:(np.zeros((image.height,image.width)),.2),'stub')
   m=self.maps(im);self.assertTrue(m.status.startswith('fallback:model_uncertain'));self.assertEqual(m.provider,'heuristic-saliency')
  finally:P.register_subject_model(None)
  s=json.loads((ROOT/'output/crello/designs/b44ab672801fd912f6c2.json').read_text())['source'] if (ROOT/'output/crello/designs/b44ab672801fd912f6c2.json').exists() else None
  if s is None:self.skipTest('Local Crello collection not installed')
  status={}
  for d in propose_photo_scene(s,{'width':1280,'height':720},analyze(s),mode='subject_aware',status=status):
   self.assertTrue(passes(validate(s,d)))
   for old,new in zip(s['elements'],d['elements']):
    self.assertLessEqual(set(new),set(old)|{'fit'})
    self.assertEqual(old.get('asset_id'),new.get('asset_id'))
  self.assertEqual(status['maps']['status'],'fallback:model_unavailable')


class PhotoSupportSafetyTests(unittest.TestCase):
 def test_transparent_host_box_does_not_support_copy_even_if_graph_claims_it(self):
  import tempfile
  from types import SimpleNamespace
  from unittest.mock import patch
  from PIL import Image
  import numpy as np
  from engine import photo_analysis as P
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); rng=np.random.default_rng(4)
   pixels=rng.integers(0,255,(100,200,4),dtype=np.uint8);pixels[:,:,3]=255
   Image.fromarray(pixels,'RGBA').save(root/'photo.png')
   pixels[:,:100,3]=0;Image.fromarray(pixels,'RGBA').save(root/'overlay.png')
   source={'canvas':{'width':200,'height':100},'elements':[
    {'id':eid,'type':'image','asset_id':asset,'fit':'fill','x':0,'y':0,'width':200,'height':100,'z_index':z}
    for eid,asset,z in [('photo','photo.png',0),('overlay','overlay.png',1)]]}
   source['elements'].append({'id':'copy','type':'text','content':'Sale','x':10,'y':20,'width':60,'height':30,'z_index':2})
   A=SimpleNamespace(graph=SimpleNamespace(edges=[{'a':'overlay','b':'copy','kind':'supports_on_photo'}]))
   with patch.object(P,'ASSETS_DIR',root),patch('engine.relations.asset_stats',return_value=SimpleNamespace(kind='photo')):
    self.assertEqual(P.source_photo_supports(source,A),{})
    self.assertEqual(P.source_photo_supports(source),{'copy':'photo'})
    moved=copy.deepcopy(source);moved['elements'][1]['x']=100
    self.assertFalse(scene_defects(source,moved)['unsupported_text'])
    moved['elements'][0]['x']=100
    self.assertEqual(scene_defects(source,moved)['unsupported_text'],['copy'])

 def test_uniform_tiny_and_transparent_images_have_finite_empty_subjects(self):
  import math
  from PIL import Image
  from engine.photo_analysis import analyze_image,canvas_field
  for size,color in [((1,1),(0,0,0,255)),((30,20),(240,170,120,255)),((1,30),(0,0,0,0))]:
   m=analyze_image(Image.new('RGBA',size,color),'flat-'+str(size))
   self.assertTrue(math.isfinite(m.confidence));self.assertFalse(m.subject.any())
   f=canvas_field({'x':0,'y':0,'width':100,'height':100,'fit':'fill','opacity':0},m,100,100)
   self.assertFalse(f.subject.any());self.assertEqual(float(f.coverage.max()),0)

 def test_cancelled_subject_scene_records_incomplete_search(self):
  from engine.edit_operators import SearchBudget
  path=ROOT/'output/crello/designs/72633b24108dde39d197.json'
  if not path.exists():self.skipTest('Local Crello collection not installed')
  source=json.loads(path.read_text())['source'];status={}
  out=propose_photo_scene(source,{'width':400,'height':100},analyze(source),mode='subject_aware',status=status,
                          budget=SearchBudget(cancelled=lambda:True))
  self.assertEqual(status['status'],'partial:cancelled');self.assertEqual(status['attempted_configs'],0)
  self.assertEqual(len(out),len(propose_photo_scene(source,{'width':400,'height':100},analyze(source),mode='heuristic')))


if __name__=='__main__':unittest.main()
