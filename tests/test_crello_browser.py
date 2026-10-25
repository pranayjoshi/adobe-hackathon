import io,json,pickle,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from service.app import app
from tools.build_crello_catalog import convert_row, PrimitiveUnpickler
from renderer.converter.brand_fonts import build_font_dict

class CrelloTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.features={k:{'feature':{'names':v}} for k,v in {'type':['ImageElement','TextElement'],'font':['','Example Sans'],'text_align':['','left']}.items()}
  b=io.BytesIO();Image.new('RGBA',(4,4),(20,40,60,128)).save(b,format='PNG')
  self.row={'id':'id/with/unsafe/path','canvas_width':100,'canvas_height':200,'length':2,'left':[0,4],'top':[0,8],'width':[100,90],'height':[200,50],'type':[0,1],'image':[{'bytes':b.getvalue()},None],'text':['','A\\nB'],'font':[0,1],'font_size':[0,20],'text_align':[0,1],'font_bold':[[],[True,True,False]],'font_italic':[[],[False,False,True]],'text_color':[[],['rgba(1,2,3,1)']*3]}
 def tearDown(self):self.tmp.cleanup()
 def test_conversion_preserves_layers_alpha_and_character_styles(self):
  key,d,notes=convert_row(self.row,self.features,self.root)
  self.assertEqual(len(d['elements']),2);self.assertEqual(d['elements'][1]['content'],'A\nB')
  self.assertEqual(d['elements'][1]['font_family'],'Example Sans')
  self.assertEqual(d['elements'][1]['style_runs'][0]['end'],1)
  self.assertEqual(d['elements'][1]['style_runs'][1]['start'],2)
  path=self.root/d['elements'][0]['asset_id'];self.assertTrue(path.resolve().is_relative_to(self.root.resolve()));self.assertEqual(Image.open(path).getpixel((0,0))[3],128)
  self.assertEqual(d['elements'][0]['fit'],'fill');self.assertTrue(notes)
 def test_missing_original_image_rejected(self):
  self.row['image'][0]=None
  with self.assertRaisesRegex(ValueError,'Missing original image'):convert_row(self.row,self.features,self.root)
 def test_unknown_font_category_rejected(self):
  self.row['font'][1]=999
  with self.assertRaisesRegex(ValueError,'Invalid font'):convert_row(self.row,self.features,self.root)
 def test_font_archive_disallows_executable_objects(self):
  self.assertEqual(PrimitiveUnpickler(io.BytesIO(pickle.dumps({'a':[b'font']}))).load(),{'a':[b'font']})
  with self.assertRaises(ValueError):PrimitiveUnpickler(io.BytesIO(pickle.dumps(Path('x')))).load()
 def test_font_manifest_rejects_traversal(self):
  (self.root/'crello-fonts.json').write_text(json.dumps([{'file':'../outside.ttf','fontFamily':'Bad','fontWeight':'regular','fontStyle':'regular'}]))
  with self.assertRaisesRegex(ValueError,'outside'):build_font_dict(str(self.root))
 def test_catalog_filters_pagination_and_scoped_media(self):
  entries=[{'id':f'{i:020x}','source_id':str(i),'title':title,'format':fmt,'width':w,'height':h,'status':status,'fonts':['Example Sans']} for i,(title,fmt,w,h,status) in enumerate([('Coffee Sale','Instagram',100,100,'ready'),('Coffee Story','Story',100,200,'needs-review'),('Tools','Banner',200,100,'ready')])]
  (self.root/'catalog.json').write_text(json.dumps({'entries':entries,'revision':'test','rejected':[]}))
  with patch('service.crello.CATALOG_DIR',self.root):
   c=TestClient(app)
   self.assertEqual(c.get('/api/crello?q=coffee').json()['total'],2)
   self.assertEqual(c.get('/api/crello?orientation=portrait&status=needs-review').json()['items'][0]['title'],'Coffee Story')
   self.assertEqual(c.get('/api/crello?format=Banner').json()['total'],1)
   self.assertEqual(c.get('/api/crello?offset=1&limit=1').json()['items'][0]['title'],'Coffee Story')
   self.assertEqual(c.get('/api/crello?limit=101').status_code,422)
   self.assertEqual(c.get('/api/crello/not-in-manifest').status_code,404)
   self.assertEqual(c.get('/api/crello/media/not-in-manifest.png').status_code,404)
   (self.root/'designs').mkdir();(self.root/'designs'/f"{entries[0]['id']}.json").write_text(json.dumps({'source':{'elements':[{'id':'x','type':'image','asset_id':'../outside.png'}]}}))
   self.assertEqual(c.get('/api/crello/'+entries[0]['id']).status_code,409)
 def test_background_images_keep_proportions_in_new_shapes(self):
  from engine.canonical import backdrop_patch,rotated_extent
  for rotation in (0,30):
   e={"type":"image","fit":"fill","width":100,"height":200,"x":0,"y":0,"rotation":rotation}
   out={**e,**backdrop_patch(e,400,100)};self.assertAlmostEqual(out["width"]/out["height"],.5)
   b=rotated_extent(out);self.assertGreaterEqual(b.x,-.01);self.assertLessEqual(b.x1,400.01);self.assertLessEqual(b.y1,100.01)
 def test_contrast_handles_extreme_negative_tracking(self):
  from engine.quality import text_contrast
  e={"id":"text","type":"text","x":50,"y":20,"width":50,"height":30,"content":"HELLO","font_family":"Montserrat","font_size":10,"letter_spacing":-500,"text_align":"right","color":"rgba(0,0,0,1)"}
  ratios,busy=text_contrast({"canvas":{"width":100,"height":100},"elements":[e]});self.assertIn("text",ratios);self.assertEqual(busy["text"],0)
 def test_capitalized_text_metrics_match_renderer_widths(self):
  from engine.typography import measure,_FM
  from renderer.converter.wrap import char_widths
  e={"content":"lowercase","font_family":"Montserrat","capitalize":True,"font_size":20}
  fit=measure(e,20,1000)
  if _FM:self.assertAlmostEqual(fit.line_width,sum(char_widths(_FM,"Montserrat","regular","regular",20,e["content"],True)))
 def test_empty_catalog_is_useful(self):
  with patch('service.crello.CATALOG_DIR',self.root):
   d=TestClient(app).get('/api/crello').json();self.assertEqual(d['total'],0);self.assertIn('importer',d['description'])

if __name__=='__main__':unittest.main()
