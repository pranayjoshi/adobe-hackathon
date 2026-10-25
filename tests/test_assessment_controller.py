"""Assessment boundary regressions, independent of optional external models."""
import copy, json, os, subprocess, tempfile, unittest, contextlib, io
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from engine import quality
from engine.evaluators import OpenJevEvaluator
from engine.judging import candidate_assessment
from engine.learning import RANK_NAMES, compatible_ranker, preference_score

def fixture():
    return {'canvas':{'width':160,'height':80},'elements':[
        {'id':'bg','type':'shape','shape_type':'rectangle','x':0,'y':0,'width':160,'height':80,'fill_color':'#000000','z_index':0},
        {'id':'first','type':'text','content':'Hi','font_family':'Solstice Sans','font_size':16,'color':'#ffffff','x':2,'y':2,'width':50,'height':24,'z_index':0},
        {'id':'veil','type':'shape','shape_type':'rectangle','x':0,'y':0,'width':160,'height':80,'fill_color':'#ffffff','opacity':.5,'z_index':0},
        {'id':'second','type':'text','content':'Hi','font_family':'Solstice Sans','font_size':16,'color':'#ffffff','x':90,'y':40,'width':50,'height':24,'z_index':0},
        {'id':'above','type':'shape','shape_type':'rectangle','x':0,'y':0,'width':160,'height':80,'fill_color':'#000000','z_index':0}]}

class AssessmentTests(unittest.TestCase):
    def test_resource_mutation_during_render_or_cached_copy_fails_closed(self):
        design=fixture();quality._CACHE.clear()
        def fake_worker(command,**kwargs):
            Image.new('RGB',(160,80),'black').save(command[4])
            Path(command[6]).write_text(json.dumps({'warnings':[]}))
            return SimpleNamespace(returncode=0)
        with patch('engine.quality.resource_fingerprint',side_effect=['before','after']),patch('engine.quality.subprocess.run',side_effect=fake_worker):
            result=quality.verify_design(design)
        self.assertFalse(result['verified']);self.assertIn('Resources changed during verification',result['error']);self.assertFalse(quality._CACHE)
        with patch('engine.quality.resource_fingerprint',return_value='before'),patch('engine.quality.subprocess.run',side_effect=fake_worker):
            self.assertTrue(quality.verify_design(design)['verified'])
        with patch('engine.quality.resource_fingerprint',side_effect=['before','after']),patch('engine.quality.subprocess.run') as worker:
            result=quality.verify_design(design)
        worker.assert_not_called();self.assertTrue(result['cache_hit']);self.assertFalse(result['verified']);self.assertIsNone(result['image'])

    def test_minority_rich_color_cannot_be_diluted_by_other_colors(self):
        from engine.analysis import analyze
        source={'canvas':{'width':200,'height':100},'elements':[
            {'id':'black','type':'shape','shape_type':'rectangle','x':0,'y':0,'width':200,'height':100,'fill_color':'#000','z_index':0},
            {'id':'red','type':'shape','shape_type':'rectangle','x':0,'y':60,'width':40,'height':40,'fill_color':'#f00','z_index':1},
            {'id':'copy','type':'text','content':'ABCDEFGHIJ','font_family':'Solstice Sans','font_size':18,'color':'#fff','x':0,'y':5,'width':180,'height':40,'z_index':2,
             'style_runs':[{'start':0,'end':0,'color':'rgba(255,0,0,.75)'}]}]}
        changed=copy.deepcopy(source);changed['elements'][1].update(y=0,height=50)
        before,_=quality.text_contrast(source);after,_=quality.text_contrast(changed)
        self.assertGreater(before['copy'],3);self.assertEqual(after['copy'],1.)
        report=quality.diagnose(source,changed,analyze(source))
        self.assertIn('copy',report['contrast_regressions'])
        # A fully overridden base color has no painted glyphs and must not
        # invent a contrast failure of its own.
        source['elements'][2].update(color='#000',style_runs=[{'start':0,'end':9,'color':'#fff'}])
        ratio,_=quality.text_contrast(source);self.assertGreater(ratio['copy'],20)

    def test_prefix_verification_honors_shared_render_budget_and_refunds_cache(self):
        from engine.analysis import analyze
        from engine.contracts import Budget
        design=fixture();A=analyze(design);image=Image.new('RGB',(160,80),'black')
        budget=Budget(seconds=5,max_renders=1)
        with patch('engine.quality.rendered_background',return_value=(image,'official')) as render:
            result=quality.diagnose(design,design,A,pixels=True,budget=budget)
        self.assertEqual(render.call_count,1);self.assertEqual(budget.renders,1)
        self.assertFalse(result['contrast_verified']);self.assertTrue(any(row['reason']=='render_budget' for row in result['verification_incomplete']))
        budget=Budget(seconds=5,max_renders=1)
        with patch('engine.quality.rendered_background',return_value=(image,'cached')):
            result=quality.diagnose(design,design,A,pixels=True,budget=budget)
        self.assertEqual(budget.renders,0);self.assertTrue(result['contrast_verified'])
        budget=Budget(seconds=5,max_renders=3,cancellation=lambda:True)
        with patch('engine.quality.rendered_background') as render:
            result=quality.diagnose(design,design,A,pixels=True,budget=budget)
        render.assert_not_called();self.assertFalse(result['contrast_verified'])

    def test_huge_layer_fails_before_bitmap_allocation(self):
        design=fixture();design['elements'][0]['width']=1e12;design['elements'][0]['height']=1e12
        quality._ESTIMATES.clear()
        original=Image.new
        def bounded_new(mode,dimensions,*args,**kwargs):
            self.assertLessEqual(dimensions[0]*dimensions[1],32_000_000)
            return original(mode,dimensions,*args,**kwargs)
        with patch('engine.quality.Image.new',side_effect=bounded_new):
            with self.assertRaisesRegex(ValueError,'allocation limit'):quality.composite(design)

    def test_missing_rich_face_cannot_be_verified_with_font_fallback(self):
        from renderer.converter.convert import font_face_warnings
        class Fonts:
            _fonts={'Solstice Sans':[{'fontWeight':'regular','fontStyle':'regular'}]}
        design=fixture();design['elements']=[design['elements'][1]]
        # A rich italic span needs its own available original face even when
        # the regular base face exists. Non-BMP characters stay in the arrays.
        example={'text':['A😀'],'font':['Solstice Sans'],'font_bold':[[False,False]],'font_italic':[[False,True]]}
        missing=font_face_warnings(design,example,Fonts())
        self.assertEqual(missing,['Original font face unavailable: Solstice Sans / regular / italic'])
        Fonts._fonts['Solstice Sans'].append({'fontWeight':'regular','fontStyle':'italic'})
        self.assertFalse(font_face_warnings(design,example,Fonts()))
        design['elements'][0]['font_family']='Unavailable Face'
        # Exercise the real isolated worker: a fallback PNG must not pass.
        result=quality.verify_design(design)
        self.assertFalse(result['verified']);self.assertIn('Original font face unavailable',result['error'])

    def test_equal_z_background_prefix_and_transparency(self):
        design=fixture()
        self.assertEqual([e['id'] for e in quality.painted_prefix(design,'first')['elements']],['bg'])
        self.assertEqual([e['id'] for e in quality.painted_prefix(design,'second')['elements']],['bg','first','veil'])
        ratios,_=quality.text_contrast(design)
        self.assertGreater(ratios['first'],20);self.assertLess(ratios['second'],4.1);self.assertGreater(ratios['second'],3.8)
        prefixes=[]
        def renderer(doc,**kwargs):
            prefixes.append([e['id'] for e in doc['elements']])
            image,_=quality.composite(doc)
            return {'verified':True,'image':image,'cache_hit':False}
        with patch('engine.quality.verify_design',side_effect=renderer):
            image,status=quality.rendered_background(design,'second')
        self.assertEqual(prefixes,[['bg','first','veil']]);self.assertEqual(status,'official')
        self.assertEqual(image.getpixel((110,50)),(128,128,128))

    def test_resource_byte_change_and_render_failure_are_not_stale_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);assets=root/'assets';fonts=root/'fonts';assets.mkdir();(fonts/'nested').mkdir(parents=True)
            Image.new('RGB',(2,2),'red').save(assets/'a.png');(fonts/'nested'/'a.ttf').write_bytes(b'font-one')
            design={'canvas':{'width':10,'height':10},'elements':[{'id':'i','type':'image','asset_id':'a.png','x':0,'y':0,'width':10,'height':10}]}
            with patch.object(quality,'ASSETS_DIR',assets),patch.object(quality,'FONTS_DIR',fonts):
                first=quality.resource_fingerprint(design)
                (fonts/'nested'/'a.ttf').write_bytes(b'font-two');second=quality.resource_fingerprint(design)
                self.assertNotEqual(first,second)
                Image.new('RGB',(2,2),'blue').save(assets/'a.png');self.assertNotEqual(second,quality.resource_fingerprint(design))
                quality._CACHE.clear()
                with patch('engine.quality.subprocess.run',side_effect=subprocess.TimeoutExpired('render',.1)):
                    result=quality.verify_design(design,timeout=.1)
                self.assertFalse(result['verified']);self.assertIsNone(result['image']);self.assertIn('timed out',result['error'])
                self.assertFalse(quality._CACHE)

    def test_pair_orientation_uncertainty_and_unavailable(self):
        design=fixture();after=copy.deepcopy(design);after['elements'][1]['x']=3
        images=[Image.new('RGB',(10,10)) for _ in range(3)]
        replies=iter([{'verdict':{'choice':'B','probabilities':{'B':.8,'A':.2}}},{'verdict':{'choice':'A','probabilities':{'A':.8,'B':.2}}}])
        evaluator=OpenJevEvaluator(transport=lambda *a,**k:next(replies))
        verdict=evaluator.compare(design,design['canvas'],design,after,images=images)
        self.assertTrue(verdict['available']);self.assertEqual(verdict['verdict'],'after');self.assertFalse(verdict['provenance']['probabilities_calibrated'])
        replies=iter([{'verdict':{'choice':'B'}},{'verdict':{'choice':'B'}}])
        verdict=evaluator.compare(design,design['canvas'],design,after,images=images)
        self.assertEqual(verdict['verdict'],'unsure')
        def unavailable(*args,**kwargs):raise TimeoutError('no server')
        result=OpenJevEvaluator(transport=unavailable).compare(design,design['canvas'],design,after,images=images)
        self.assertFalse(result['available']);self.assertEqual(result['verdict'],'unavailable')

    def test_invalid_high_score_and_incompatible_ranker_cannot_pass(self):
        invalid={'valid':False,'score':{'overall':99999},'rubric':{'gate':False}}
        self.assertFalse(candidate_assessment(invalid,{'verified':True})['eligible'])
        valid={'valid':True,'rubric':{'gate':True,'tier2':{}}}
        self.assertFalse(candidate_assessment(valid,{'verified':False})['eligible'])
        model={'features':RANK_NAMES,'weights':[1.]*len(RANK_NAMES),'mean':[0.]*len(RANK_NAMES),'scale':[1.]*len(RANK_NAMES)}
        self.assertTrue(compatible_ranker(model))
        for field,value in [('weights',[1.]),('scale',[0.]*len(RANK_NAMES)),('weights',[float('nan')]*len(RANK_NAMES)),('report',{'deployable':False}),('report',[])]:
            broken={**model,field:value};self.assertFalse(compatible_ranker(broken))
            with patch('engine.learning.load_model',return_value=broken):self.assertIsNone(preference_score({}))

    def test_fitted_export_agrees_with_runtime_and_stays_unpromoted(self):
        from training.train_ranker import train
        from engine.learning import rank_features
        def candidate(readability):
            return {'valid':True,'score':{'parts':{'readability':readability,'hierarchy':70,'composition':70,'fidelity':100},'stats':{'coverage':.5}},
                    'diagnostics':{'overflow':[],'collisions':[],'contrast_regressions':[]},'rubric':{'gate':True,'tier2':{'scores':{},'stats':{},'warnings':[]}}}
        rows=[{'family':'f'+str(n%5),'kind':'human','decision':'a','a':candidate(85+n%3),'b':candidate(45+n%3)} for n in range(25)]
        manifest={'schema_version':1,'families':{'f'+str(n):'test' if n==4 else 'train' for n in range(5)}}
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            model=train(rows,out=Path(folder)/'ranker.json',split_manifest=manifest)
        self.assertFalse(model['report']['deployable'])
        with patch('engine.learning.load_model',return_value=model):self.assertIsNone(preference_score(rows[0]['a']))
        # Simulate a promoted artifact only inside this runtime parity test.
        model['report']['deployable']=True;features=rank_features(rows[0]['a'])
        expected=sum(w*(x-mu)/sd for w,x,mu,sd in zip(model['weights'],features,model['mean'],model['scale']))
        with patch('engine.learning.load_model',return_value=model):self.assertAlmostEqual(preference_score(rows[0]['a']),expected)
        self.assertEqual(model['report']['held_out_human_accuracy'],1.)

if __name__=='__main__':unittest.main()
