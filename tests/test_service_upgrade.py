import copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from service.app import app
from engine.pipeline import generate
ROOT=Path(__file__).resolve().parents[1]
class ServiceTests(unittest.TestCase):
 def setUp(self):
    self.tmp=tempfile.TemporaryDirectory();self.store=patch('service.workspace.STORE',Path(self.tmp.name));self.store.start();self.c=TestClient(app)
    self.task=json.loads(next((ROOT/'input').glob('*level1*.json')).read_text())
 def tearDown(self):self.store.stop();self.tmp.cleanup()
 def request(self):return {'source':self.task['source'],'targets':[self.task['target_canvas']],'options':{'render':False,'solver':False}}
 def test_generation_history_and_human_pair(self):
    r=self.c.post('/api/generate',json=self.request());self.assertEqual(r.status_code,200,r.text);d=r.json();self.assertTrue(d['run_id']);self.assertEqual(len(self.c.get('/api/history').json()),1)
    pair=self.c.get('/api/preferences/pair').json();self.assertIsNotNone(pair)
    self.assertEqual(self.c.post('/api/preferences',json={**pair,'decision':'a'}).status_code,200)
    row=json.loads((Path(self.tmp.name)/'labels.jsonl').read_text());self.assertEqual(row['kind'],'human');self.assertTrue(row['a']['diagnostics'])
 def test_bad_canvas_is_422(self):
    r=self.request();r['source']['canvas']['width']=0;self.assertEqual(self.c.post('/api/generate',json=r).status_code,422)
 def test_unknown_lock_is_422(self):
    r=self.request();r['locks']={'missing':{'x':0,'y':0}};self.assertEqual(self.c.post('/api/generate',json=r).status_code,422)
 def test_intent_is_validated_persisted_and_kept_outside_source(self):
    r=self.request();r['options'].update(group_policies='hierarchical',refine=False)
    bad={**r,'analysis_overrides':{'roles':{'missing':'headline'}}}
    self.assertEqual(self.c.post('/api/jobs',json=bad).status_code,422)
    eid=max((e for e in r['source']['elements'] if e['type']=='text'),key=lambda e:e['font_size'])['id']
    original=copy.deepcopy(r['source']);r['analysis_overrides']={'roles':{eid:'body'}}
    generated=self.c.post('/api/generate',json=r);self.assertEqual(generated.status_code,200,generated.text)
    data=generated.json();self.assertEqual(data['analysis']['roles'][eid],'body')
    saved=self.c.get('/api/history/'+data['run_id']).json()['request']
    self.assertEqual(saved['analysis_overrides'],r['analysis_overrides'])
    self.assertEqual(saved['source'],original)
 def test_job_completes_and_persists(self):
    r=self.c.post('/api/jobs',json=self.request());self.assertEqual(r.status_code,200)
    for _ in range(100):
        job=self.c.get('/api/jobs/'+r.json()['id']).json()
        if job['state'] not in ('queued','running'):break
        time.sleep(.05)
    self.assertEqual(job['state'],'done',job);self.assertEqual(job['progress'],1)
 def test_asset_traversal_rejected(self):
    d=self.task['source'];e=next(e for e in d['elements'] if e['type']=='image');e['asset_id']='../fonts/not-an-image'
    self.assertEqual(self.c.post('/api/express/assets',json={'design':d}).status_code,422)
 def test_record_path_traversal_rejected(self):
    self.assertNotEqual(self.c.get('/api/history/..%2Fsecrets').status_code,200)
if __name__=='__main__':unittest.main()
