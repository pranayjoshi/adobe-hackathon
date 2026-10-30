import hashlib,json,tempfile,unittest
from pathlib import Path
from evaluation.visual_suite import case_matrix, complete_slot, load_source, atomic_json, digest

class VisualSuiteTests(unittest.TestCase):
    def test_exact_matrix_ignores_embedded_targets(self):
        suite=json.loads(Path('docs/individual-update-plans/visual-suite.v1.json').read_text())
        cases=case_matrix(suite)
        self.assertEqual(16,len(cases));self.assertEqual(4,len({c['format']['id'] for c in cases}))
        suite['expected_cases_per_arm']=17
        with self.assertRaises(ValueError):case_matrix(suite)

    def test_source_hash_and_pure_envelope(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source.json';source={'canvas':{'width':1,'height':2},'elements':[],'metadata':{'private':'provenance'}}
            atomic_json(p,{'source':source,'target_canvas':{'width':999,'height':999}})
            spec={'id':'x','path':'source.json','sha256':digest(p),'source_canvas':source['canvas'],'element_count':0}
            self.assertEqual({'canvas','elements'},set(load_source(spec,td)))
            p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_source(spec,td)

    def test_failed_and_resumable_artifact_completeness(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);atomic_json(p/'status.json',{'status':'failed'})
            self.assertFalse(complete_slot(p));atomic_json(p/'failure.json',{'error':'missing asset'})
            self.assertTrue(complete_slot(p))
            artifacts={}
            for f in ('final.json','final.png','assessment.json','trace.jsonl'):
                (p/f).write_text(f);artifacts[f]=digest(p/f)
            atomic_json(p/'status.json',{'status':'complete','artifacts':artifacts})
            self.assertTrue(complete_slot(p));(p/'final.png').write_text('corrupt')
            self.assertFalse(complete_slot(p))

class FamilyIsolationTests(unittest.TestCase):
    def test_resizes_trajectories_and_exact_content_share_partition(self):
        from training.families import build_split_manifest,split_of
        el={'id':'a','type':'text','content':'Same copy','font_family':'A','color':'#000','font_size':20,'x':0,'y':0,'width':50,'height':20}
        records=[('first',{'family':'campaign-one','source':{'elements':[el]}}),('second',{'family':'campaign-two','source':{'elements':[{**el,'id':'different','font_size':30,'width':100}]}}),('trajectory',{'source_family':'campaign-two','source':{'elements':[el]}})]
        manifest=build_split_manifest(records)
        self.assertEqual(split_of('campaign-one',manifest),split_of('campaign-two',manifest))
        self.assertEqual(split_of('first',manifest),split_of('trajectory',manifest))
        with self.assertRaises(ValueError):split_of('unknown',manifest)
        manifest['aliases']={'a':'b','b':'a'}
        with self.assertRaisesRegex(ValueError,'cyclic'):split_of('a',manifest)

    def test_ambiguous_duplicate_elements_are_quarantined(self):
        from training.families import match
        el={'id':'a','type':'text','content':'same','font_family':'A','color':'#000'}
        design={'canvas':{'width':1,'height':1},'elements':[el,{**el,'id':'b'}]}
        self.assertIsNone(match(design,design))

class ExportLeakageTests(unittest.TestCase):
    def test_reviewed_visual_family_never_exported_even_when_samples_allowed(self):
        from training.export import export
        from unittest.mock import patch
        source_path=Path('output/crello/designs/b44ab672801fd912f6c2.json')
        if not source_path.exists(): self.skipTest('local Crello panel unavailable')
        record=json.loads(source_path.read_text()); source=record['source']
        # A related resize changes IDs and geometry but keeps immutable content.
        for index,e in enumerate(source['elements']):e['id']='resized-'+str(index);e['x']=float(e['x'])+1
        with tempfile.TemporaryDirectory() as td:
            corpus=Path(td)/'corpus';corpus.mkdir();atomic_json(corpus/'related.json',record)
            with patch('training.export._image',side_effect=AssertionError('reviewed panel reached renderer')):
                report=export(corpus,Path(td)/'export',allow_samples=True)
            self.assertFalse(report['counts'].get('train/sft'))
            self.assertFalse(report['counts'].get('validation/sft'))

class EvidenceAccountingTests(unittest.TestCase):
    def test_paired_report_keeps_failed_cases_and_rejects_missing_slots(self):
        from evaluation.compare_visual_runs import compare
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b'
            manifest={'suite_sha256':'same','resources':{'sha256':'same'}}
            failed={'case':'failure','family':'f','status':'failed','valid':False,'rendered':False}
            good={'case':'good','family':'f','status':'complete','valid':True,'rendered':True,'score':80,'seconds':1}
            for folder in (a,b):
                atomic_json(folder/'manifest.json',manifest)
                atomic_json(folder/'results.json',{'summary':{'failed':1},'cases':[failed,good]})
            report=compare(a,b)
            self.assertEqual(2,report['summary']['cases']);self.assertEqual(1,report['summary']['paired_scored'])
            self.assertIsNone(report['summary']['human_win_tie_loss'])
            atomic_json(b/'results.json',{'summary':{'failed':0},'cases':[good]})
            with self.assertRaisesRegex(ValueError,'case sets'):compare(a,b)

    def test_release_requires_actual_artifacts_and_same_case_set(self):
        from evaluation.release_gate import check
        with tempfile.TemporaryDirectory() as td:
            a,b=Path(td)/'a',Path(td)/'b'
            summary={'cases':1,'tier0_failures':0,'tier1_failures':{},'offline':True,'network_attempts':[], 'pixels':True,'render_failures':0,'tier2_by_bucket':{},'corpus_hash':'one'}
            case={'case':'one','rubric':{'defects':[]},'rendered':True,'preview':'missing.png'}
            atomic_json(a/'results.json',{'summary':summary,'cases':[case]})
            atomic_json(b/'results.json',{'summary':{**summary,'corpus_hash':'two'},'cases':[{**case,'case':'two'}]})
            checks={r['check']:r for r in check(a,b)}
            self.assertFalse(checks['Required image and editable artifacts']['passed'])
            self.assertFalse(checks['Paired identical cases']['passed'])
            self.assertFalse(checks['Paired frozen corpus']['passed'])

class PreferenceExportTests(unittest.TestCase):
    def test_human_pair_and_committed_correction_orientation_and_exclusion(self):
        from tools.export_preferences import export_preferences
        from service.storage import Repository
        from engine.contracts import design_hash
        from training.families import immutable_signature
        from PIL import Image
        from unittest.mock import patch
        source={'canvas':{'width':100,'height':100},'elements':[{'id':'s','type':'shape','shape_type':'rectangle','fill_color':'#ff0000','x':10,'y':10,'width':40,'height':40}]}
        after=json.loads(json.dumps(source));after['elements'][0]['x']=15
        with tempfile.TemporaryDirectory() as td:
            repo=Repository(Path(td)/'workspace');rid='run';a_hash,b_hash=design_hash(source),design_hash(after)
            repo.put_run({'id':rid,'created':1,'family':'campaign','request':{'source':source},'response':{}})
            for hashed,design,parent,operation in ((a_hash,source,None,'seed'),(b_hash,after,a_hash,'human_correction')):
                repo.put_revision(rid,0,{'design':design,'verified':True,'render':'/verified/'+hashed,'accepted':True,'parent_revision':parent,'operation':operation})
                path=repo.root/rid/'0'/(hashed+'.png');path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(100,100)).save(path)
            context={'family':'campaign','source_hash':a_hash,'target':source['canvas'],'options':{'controller_mode':'single'}}
            repo.feedback(rid,0,a_hash,'b',other_hash=b_hash,context=context,reviewer_id='reviewer-1')
            repo.feedback(rid,0,b_hash,'correction',other_hash=a_hash,context=context,reviewer_id='reviewer-1')
            repo.feedback(rid,0,a_hash,'like',context=context)
            repo.feedback(rid,0,a_hash,'a',other_hash=b_hash,context=context,provenance='model')
            output=Path(td)/'pairs.jsonl';report=export_preferences(repo.root,output)
            rows=[json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(2,len(rows));self.assertEqual('b',rows[0]['decision']);self.assertEqual(a_hash,rows[0]['a_hash']);self.assertEqual(b_hash,rows[0]['b_hash'])
            self.assertEqual('a',rows[1]['decision']);self.assertEqual(b_hash,rows[1]['a_hash']);self.assertEqual(a_hash,rows[1]['b_hash'])
            self.assertEqual('reviewer-1',rows[0]['reviewer_id']);self.assertEqual('human',rows[1]['provenance']);self.assertEqual(context,rows[0]['context'])
            self.assertEqual(rows[0]['split'],rows[1]['split']);self.assertFalse(report['training_performed'])
            with patch('tools.export_preferences.reviewed_panel_signatures',return_value={immutable_signature(source)}):
                report=export_preferences(repo.root,Path(td)/'excluded.jsonl')
            self.assertEqual(0,report['counts'].get('exported',0));self.assertEqual(2,report['counts']['reviewed_visual_panel'])
            (repo.root/'QA_ONLY').write_text('agent UI verification only')
            with self.assertRaisesRegex(ValueError,'QA_ONLY'):export_preferences(repo.root,Path(td)/'forbidden.jsonl')

    def test_training_corpus_exporters_refuse_qa_workspaces(self):
        from training.edit_dataset import collect
        from training.export import export
        from training.families import mine
        from training.perturb import run
        with tempfile.TemporaryDirectory() as td:
            corpus=Path(td)/'workspace';corpus.mkdir();(corpus/'QA_ONLY').write_text('agent UI verification only')
            atomic_json(corpus/'design.json',{'canvas':{'width':10,'height':10},'elements':[{'id':'t','type':'text'}]})
            for call in (lambda:export(corpus,Path(td)/'export'),lambda:collect(corpus,Path(td)/'demos.jsonl',[{'width':5,'height':5}]),
                         lambda:mine(corpus),lambda:run(corpus,Path(td)/'perturb')):
                with self.assertRaisesRegex(ValueError,'QA_ONLY'):call()
            self.assertFalse((Path(td)/'export').exists());self.assertFalse((Path(td)/'demos.jsonl').exists())
