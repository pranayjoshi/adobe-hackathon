import copy,json,math,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from engine.analysis import analyze
from engine.pipeline import generate
from engine.invariants import passes,validate
from engine.quality import diagnose,text_contrast
from engine.repair import repair
from engine.constraints import solve
from engine.retrieval import retrieve,export_layout
from engine.learning import apply_roles,FEATURE_NAMES

ROOT=Path(__file__).resolve().parents[1]

def sample(level=2):return json.loads(next((ROOT/'input').glob(f'*level{level}*.json')).read_text())

class UpgradeTests(unittest.TestCase):
 def test_samples_preserve_content_and_fit(self):
    for level in (1,2,3):
        t=sample(level);r=generate(t['source'],t['target_canvas'],options={'render':False},top_k=1)[0]
        self.assertTrue(r['valid'],r['checks']);self.assertFalse(r['diagnostics']['overflow'])
        self.assertEqual([e['id'] for e in t['source']['elements']],[e['id'] for e in r['design']['elements']])
 def test_no_input_mutation(self):
    t=sample(1);before=copy.deepcopy(t);generate(t['source'],t['target_canvas'],options={'render':False,'solver':False});self.assertEqual(t,before)
 def test_duplicate_ids_fail(self):
    t=sample(1)['source'];out=copy.deepcopy(t);out['elements'].append(copy.deepcopy(out['elements'][0]));self.assertFalse(passes(validate(t,out)))
 def test_nonfinite_text_fails(self):
    t=sample(1)['source'];out=copy.deepcopy(t);next(e for e in out['elements'] if e['type']=='text')['font_size']=float('nan');self.assertFalse(passes(validate(t,out)))
 def test_opaque_ids_do_not_affect_geometry(self):
    t=sample(1);a=generate(t['source'],t['target_canvas'],options={'render':False,'solver':False},top_k=1)[0]['design'];renamed=copy.deepcopy(t)
    for i,e in enumerate(renamed['source']['elements']):e['id']=f'opaque_{i}'
    b=generate(renamed['source'],renamed['target_canvas'],options={'render':False,'solver':False},top_k=1)[0]['design']
    for x,y in zip(a['elements'],b['elements']):self.assertEqual({k:v for k,v in x.items() if k!='id'},{k:v for k,v in y.items() if k!='id'})
 def test_identity_preserves_geometry(self):
    s=sample(1)['source'];r=generate(s,s['canvas'],options={'render':False},top_k=1)[0]
    self.assertEqual(s,r['design'])
 def test_long_word_fits_without_changing_content(self):
    t=sample(1);e=next(e for e in t['source']['elements'] if e['type']=='text');e['content']='X'*200
    r=generate(t['source'],{'width':728,'height':90},options={'render':False},top_k=1)[0];self.assertTrue(r['valid']);self.assertEqual(next(x for x in r['design']['elements'] if x['id']==e['id'])['content'],e['content'])
 def test_lock_placement_preserved(self):
    t=sample(1);r=generate(t['source'],t['target_canvas'],options={'render':False},top_k=1)[0];e=next(e for e in r['design']['elements'] if e['type']=='text');rect={k:e[k] for k in ('x','y','width','height')}
    result=generate(t['source'],t['target_canvas'],locks={e['id']:rect},options={'render':False},top_k=1)[0];self.assertTrue(result['valid']);self.assertEqual({k:next(x for x in result['design']['elements'] if x['id']==e['id'])[k] for k in rect},rect)
 def test_composited_shape_contrast(self):
    s={'canvas':{'width':200,'height':100},'elements':[{'id':'bg','type':'shape','shape_type':'rectangle','fill_color':'#000000','x':0,'y':0,'width':200,'height':100,'z_index':0},{'id':'t','type':'text','content':'Hello','font_family':'Solstice Sans','font_size':20,'color':'#ffffff','x':10,'y':10,'width':150,'height':40,'z_index':1}]}
    c,_=text_contrast(s);self.assertGreater(c['t'],20)
 def test_solver_removes_new_overlap(self):
    s={'canvas':{'width':500,'height':500},'elements':[{'id':str(i),'type':'text','content':'Hello','font_family':'Solstice Sans','font_size':20,'color':'#000000','x':10+i*250,'y':10,'width':100,'height':40,'z_index':i} for i in range(2)]};A=analyze(s);d=copy.deepcopy(s);d['elements'][1]['x']=10
    out,status=solve(s,d,A);self.assertIsNotNone(out,status);self.assertFalse(diagnose(s,out,A)['collisions'])
 def test_retrieval_keeps_all_ids(self):
    s=sample(1)['source'];A=analyze(s)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'layouts.json';p.write_text(json.dumps([export_layout(s,A)]))
        with patch.dict(os.environ,{'CHAMELEON_LAYOUT_LIBRARY':str(p)}):out=retrieve(s,{'width':1080,'height':1350},A)
    self.assertTrue(out);self.assertTrue(passes(validate(s,out[0])))
 def test_missing_models_neutral(self):
    s=sample(1)['source'];A=analyze(s)
    with patch.dict(os.environ,{'CHAMELEON_ROLE_MODEL':'1','CHAMELEON_ROLE_WEIGHTS':'/missing'}):self.assertEqual(apply_roles(s,A)['status'],'missing or incompatible weights')
 def test_renderer_failure_falls_back_to_estimate(self):
    s=sample(1)['source']
    with patch('engine.quality.rendered_background',return_value=(None,'renderer failed')):d=diagnose(s,s,analyze(s),pixels=True)
    self.assertEqual(d['render_mode'],'renderer failed');self.assertTrue(d['contrast'])

if __name__=='__main__':unittest.main()
