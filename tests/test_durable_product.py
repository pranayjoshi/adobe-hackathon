"""Critical local product flows, with controlled workers and cheap previews."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import os
import select
import subprocess
import sys
import json
import hashlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from service.app import app
from service.workspace import repository,record
from service.storage import Repository
from engine.analysis import analyze
from engine.contracts import design_hash
from engine.pipeline import assess_design


def source():
    return {"canvas":{"width":200,"height":100},"elements":[{
        "id":"title","type":"text","content":"Hello","font_family":"Solstice Sans",
        "font_size":20,"color":"rgba(0,0,0,1)","x":10,"y":20,"width":100,"height":40}]}


def fake_generate(design,target,*args,**kwargs):
    result=assess_design(design,target,design,analyze(design))
    result['controller']={'mode':'single','events':[],'stop_reason':'threshold','usage':{}}
    artifact=kwargs.get('artifact_callback')
    if artifact:
        artifact({'design':result['design'],'accepted':True,'verified':True,
                  'operation':'initial','image':Image.new('RGB',(target['width'],target['height']),'white')})
    return [result]


class DurableProductTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=patch('service.workspace.STORE',Path(self.temp.name));self.store.start()
        self.renderer=patch('service.app._render',return_value=('/renders/test.png',[]));self.renderer.start()
        self.generator=patch('service.app.generate',side_effect=fake_generate);self.generator.start()
        from service.app import _perform_job
        self.supervisor=patch('service.app._supervise_job',side_effect=_perform_job);self.supervisor.start()
        self.client=TestClient(app)
        self.body={'source':source(),'targets':[{'width':200,'height':100}],
                   'options':{'controller_mode':'single'}}

    def tearDown(self):
        self.supervisor.stop();self.generator.stop();self.renderer.stop();self.store.stop();self.temp.cleanup()

    def test_edit_feedback_identity_undo_and_stale_compare_and_swap(self):
        generated=self.client.post('/api/generate',json=self.body)
        self.assertEqual(generated.status_code,200,generated.text)
        data=generated.json();rid=data['run_id'];original=data['results'][0]['candidates'][0]['revision_hash']
        endpoint=f'/api/history/{rid}/targets/0'
        edit={'expected_revision_hash':original,'patches':{'title':{'x':12}}}
        committed=self.client.post(endpoint+'/edit',json=edit)
        self.assertEqual(committed.status_code,200,committed.text)
        revised=committed.json()['revision_hash'];self.assertNotEqual(original,revised)
        self.assertEqual(self.client.post(endpoint+'/edit',json=edit).status_code,409)
        restore=self.client.post(endpoint+'/restore',json={'expected_revision_hash':revised,'revision_hash':original})
        self.assertEqual(restore.status_code,200,restore.text)
        self.assertEqual(record(rid)['response']['results'][0]['candidates'][0]['design']['elements'][0]['x'],10)
        with patch('service.app._render',return_value=(None,['font bytes changed'])):
            failed_restore=self.client.post(endpoint+'/restore',json={'expected_revision_hash':original,'revision_hash':revised})
        self.assertEqual(failed_restore.status_code,422)
        self.assertEqual(repository().active_hash(rid,0),original)
        vote={'target_index':0,'revision_hash':revised,'verdict':'like','idempotency_key':'one-vote'}
        first=self.client.post(f'/api/history/{rid}/feedback',json=vote)
        second=self.client.post(f'/api/history/{rid}/feedback',json=vote)
        self.assertEqual(first.json(),second.json())
        context=first.json()['feedback']['context'];self.assertEqual(context['source_hash'],design_hash(source()))
        self.assertEqual(first.json()['feedback']['reviewer_id'],'local')
        self.assertEqual(self.client.post(f'/api/history/{rid}/feedback',json={**vote,'verdict':'dislike'}).status_code,409)
        self.assertEqual(self.client.post(endpoint+'/edit',json={'expected_revision_hash':original,'patches':{'title':{'x':True}}}).status_code,422)
        unrendered=repository().put_revision(rid,0,{'design':{**source(),'elements':[{**source()['elements'][0],'x':15}]},'verified':False})
        self.assertEqual(self.client.post(f'/api/history/{rid}/feedback',json={**vote,'idempotency_key':'bad-preview','revision_hash':unrendered['revision_hash']}).status_code,422)

    def test_multitarget_parent_survives_cancel_and_recovery_without_late_overwrite(self):
        entered=threading.Event();release=threading.Event();finished=threading.Event();calls=[]
        def controlled(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:
                entered.set();self.assertTrue(release.wait(5))
                try:return fake_generate(*args,**kwargs)
                finally:finished.set()
            return fake_generate(*args,**kwargs)
        self.generator.stop();self.generator=patch('service.app.generate',side_effect=controlled);self.generator.start()
        request={**self.body,'targets':[{'width':200,'height':100},{'width':200,'height':100}],
                 'idempotency_key':'stable-job'}
        started=self.client.post('/api/jobs',json=request);self.assertEqual(started.status_code,200,started.text)
        rid=started.json()['id']
        self.assertTrue(entered.wait(5))
        try:
            duplicate=self.client.post('/api/jobs',json=request)
            self.assertEqual(duplicate.json()['id'],rid);self.assertFalse(duplicate.json()['created'])
            pending=self.client.get('/api/jobs/'+rid).json()
            self.assertEqual(pending['result']['run_id'],rid)
            self.assertEqual(len(pending['result']['results']),1)
            self.assertTrue(pending['active_preview']['render'])
            rev=pending['active_preview']['revision_hash']
            blocked=self.client.post(f'/api/history/{rid}/targets/0/edit',json={'expected_revision_hash':rev,'patches':{'title':{'x':12}}})
            self.assertEqual(blocked.status_code,409)
            self.assertEqual(self.client.post('/api/jobs/'+rid+'/cancel').status_code,200)
            # Simulated worker/service loss: durable state wins over a late
            # callback from a retired local worker.
            repository().interrupt_orphaned()
            recovered=Repository(Path(self.temp.name)).job(rid)
            self.assertEqual(recovered['state'],'interrupted')
            self.assertEqual(len(recovered['result']['results']),1)
        finally:
            release.set();self.assertTrue(finished.wait(5))
            # Join the executor queue deterministically before temp cleanup.
            from service.workspace import POOL
            POOL.submit(lambda:None).result(timeout=5)
        persisted=repository().job(rid)
        self.assertEqual(persisted['state'],'interrupted')
        self.assertEqual(len(record(rid)['response']['results']),1)
        self.assertEqual(self.client.post('/api/jobs/'+rid+'/resume').status_code,200)
        POOL.submit(lambda:None).result(timeout=5)
        self.assertEqual(repository().job(rid)['state'],'done')
        self.assertEqual(len(record(rid)['response']['results']),2)
        self.assertEqual(len(calls),3)  # the successful first target was reused

    def test_claim_exclusion_missing_preview_and_resource_validation(self):
        repo=repository();rid,created=repo.create_job('claimjob',self.body,'claim-key')
        self.assertTrue(created)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda owner:repo.claim(rid,owner),('one','two')))
        self.assertEqual(sum(results),1)
        repo.update_job(rid,{'state':'cancelled'})
        self.assertFalse(repo.update_job(rid,{'state':'done'}))
        def no_artifact(*args,**kwargs):
            kwargs['artifact_callback']=None
            return fake_generate(*args,**kwargs)
        with patch('service.app._render',return_value=(None,['missing font'])),patch('service.app.generate',side_effect=no_artifact):
            result=self.client.post('/api/generate',json=self.body)
        self.assertEqual(result.status_code,422,result.text)
        result_data=result.json()['detail'];self.assertEqual(result_data['results'][0]['status'],'no_verified_output')
        self.assertFalse(result_data['results'][0]['candidates'])
        self.assertFalse(repository().active_hash(result_data['run_id'],0))
        for change in ({'width':1e12},{'rotation':'bad'},{'opacity':True},{'style_runs':[{'start':True}]}):
            malformed=copy.deepcopy(self.body);malformed['source']['elements'][0].update(change)
            self.assertEqual(self.client.post('/api/jobs',json=malformed).status_code,422)
        self.assertEqual(self.client.post('/api/jobs',json={**self.body,'options':{'controller_mode':True}}).status_code,422)
        self.assertEqual(self.client.post('/api/jobs',json={**self.body,'locks':{'title':{'x':True,'y':0}}}).status_code,422)
        # Built-in Level 2 intentionally has a rich-text run beyond the final
        # character. Renderer clamps it; validation must preserve that input.
        from service.app import _check_source
        from engine.canonical import ROOT
        sample=json.loads(next((ROOT/'input').glob('*level2*.json')).read_text())
        _check_source(sample['source'])

    def test_missing_original_image_stays_visible_and_unverified(self):
        warning="Missing asset 'badge.png' for element 'mark' - rendered a placeholder."
        def no_artifact(*args,**kwargs):
            kwargs['artifact_callback']=None
            return fake_generate(*args,**kwargs)
        with patch('service.app._render',return_value=(None,['Missing or unsafe image asset: badge.png'])),patch('service.app._unverified_source_preview',return_value=('/renders/unverified-test.png',[warning])),patch('service.app.generate',side_effect=no_artifact):
            result=self.client.post('/api/generate',json=self.body)
        self.assertEqual(result.status_code,200,result.text)
        data=result.json();row=data['results'][0];candidate=row['candidates'][0]
        self.assertEqual(row['status'],'unverified_missing_asset')
        self.assertEqual(candidate['render'],'/renders/unverified-test.png')
        self.assertFalse(candidate['preview_verified']);self.assertFalse(candidate['exportable'])
        self.assertFalse(candidate['verification']['verified'])
        self.assertIsNone(repository().active_hash(data['run_id'],0))

    def test_retired_worker_cannot_mutate_revisions_events_or_parent_run(self):
        repo=repository();rid,_=repo.create_job('fencedworker',self.body)
        self.assertTrue(repo.claim(rid,'old-attempt'))
        repo.put_run({'id':rid,'created':1,'response':{'results':[]}},owner='old-attempt')
        repo.interrupt_orphaned();self.assertTrue(repo.requeue(rid));self.assertTrue(repo.claim(rid,'new-attempt'))
        self.assertIsNone(repo.put_revision(rid,0,{'design':source()},activate=True,owner='old-attempt'))
        self.assertIsNone(repo.append_event(rid,{'type':'accepted'},owner='old-attempt'))
        self.assertFalse(repo.put_run({'id':rid,'created':2,'response':{'results':['stale']}},owner='old-attempt'))
        self.assertIsNone(repo.active_hash(rid,0));self.assertEqual(repo.events(rid),[])
        saved=repo.put_revision(rid,0,{'design':source()},activate=True,owner='new-attempt')
        self.assertEqual(repo.active_hash(rid,0),saved['revision_hash'])

    def test_mixed_evaluation_reports_remain_visible_without_server_error(self):
        root=Path(self.temp.name)/'report-fixture'
        reports={'good':{'summary':{'valid':3,'cases':4}},'legacy':[],
                 'invalid-counts':{'summary':{'valid':'three','cases':4}}}
        for name,data in reports.items():
            folder=root/'output/evaluation'/name;folder.mkdir(parents=True)
            (folder/'results.json').write_text(json.dumps(data))
        broken=root/'output/evaluation/corrupt';broken.mkdir(parents=True)
        (broken/'results.json').write_text('{not json')
        with patch('service.workspace.ROOT',root):
            response=self.client.get('/api/evaluation');self.assertEqual(response.status_code,200,response.text)
            listed={row['name']:row for row in response.json()}
            self.assertEqual(listed['good']['valid'],3);self.assertTrue(listed['good']['available'])
            for name in ('legacy','invalid-counts','corrupt'):
                self.assertFalse(listed[name]['available']);self.assertTrue(listed[name]['diagnostic'])
                self.assertEqual(self.client.get('/api/evaluation/'+name).status_code,422)
            self.assertEqual(self.client.get('/api/evaluation/good').status_code,200)

    def test_corrupt_cached_preview_is_replaced_by_verified_pixels(self):
        self.renderer.stop()
        from service.app import _render
        design=source();identity=design_hash(design);resources='test-resource-hash'
        folder=Path(self.temp.name)/'render-cache';folder.mkdir()
        key=hashlib.sha256((identity+resources).encode()).hexdigest()[:24]
        destination=folder/(key+'.png');destination.write_bytes(b'broken png')
        destination.with_suffix('.json').write_text(json.dumps({'verified':True,'design_hash':identity,'resource_hash':resources}))
        verified={'verified':True,'design_hash':identity,'resource_hash':resources,'warnings':[],
                  'profile':'official-exact-v1','image':Image.new('RGB',(200,100),'white')}
        with patch('service.app.RENDER_DIR',folder),patch('engine.quality.resource_fingerprint',return_value=resources),patch('engine.quality.verify_design',return_value=verified):
            url,warnings=_render(design)
        self.assertEqual(url,'/renders/'+destination.name);self.assertFalse(warnings)
        with Image.open(destination) as image:
            image.load();self.assertEqual(image.size,(200,100))

    def test_supervisor_reaps_unresponsive_cancelled_worker(self):
        self.supervisor.stop()
        from service.app import GenerateRequest,_supervise_job
        repo=repository();rid,_=repo.create_job('blockedworker',self.body)
        self.assertTrue(repo.claim(rid,'supervisor-test'))
        ready_read,ready_write=os.pipe();real_popen=subprocess.Popen;child=[];grandchild=None
        def disposable(*args,**kwargs):
            descendant=f'import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);os.write({ready_write},("ready:"+str(os.getpid())).encode());signal.pause()'
            leader=f'import subprocess,sys,signal;subprocess.Popen([sys.executable,"-c",{descendant!r}],pass_fds=({ready_write},));signal.pause()'
            process=real_popen([sys.executable,'-c',leader],
                pass_fds=(ready_write,),start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            child.append(process);return process
        try:
            with patch('service.app.subprocess.Popen',side_effect=disposable),patch('service.app._WORKER_CANCEL_GRACE_SECONDS',.05):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future=pool.submit(_supervise_job,GenerateRequest.model_validate(self.body),rid,'supervisor-test',repo)
                    self.assertTrue(select.select([ready_read],[],[],5)[0])
                    message=os.read(ready_read,100).decode();self.assertTrue(message.startswith('ready:'))
                    grandchild=int(message.split(':')[1])
                    repo.update_job(rid,{'cancel':True})
                    future.result(timeout=5)
            self.assertEqual(repo.job(rid)['state'],'cancelled')
            self.assertIsNotNone(child[0].poll())
            with self.assertRaises(ProcessLookupError):os.kill(child[0].pid,0)
            # A killed orphan can briefly remain a zombie awaiting OS reaping;
            # it must not remain an executing renderer descendant.
            status=subprocess.run(['ps','-o','stat=','-p',str(grandchild)],capture_output=True,text=True).stdout.strip()
            self.assertTrue(not status or status.startswith('Z'),status)
        finally:
            os.close(ready_read);os.close(ready_write)
            for process in child:
                if process.poll() is None:process.kill();process.wait(timeout=5)
            if grandchild:
                try:os.kill(grandchild,9)
                except ProcessLookupError:pass


if __name__=='__main__':unittest.main()
