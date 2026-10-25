"""Lean checks for guide 15: plan validation, round-trip fidelity, dataset contract, masking, fail-closed bundles."""
import copy,json,math,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from engine.analysis import analyze
from engine.contracts import DesignRevision,design_hash
from engine.invariants import passes,validate
from engine.materialize import apply_edit
from engine.neural_proposals import (PLAN_V2,PlanRejected,action_to_proposal,apply_plan,design_to_plan,design_to_plan_v2,
                                     edit_action,materialize_plan,parse_plan_text,plan_fidelity,validate_plan)
ROOT=Path(__file__).resolve().parents[1]
TARGET={'width':1200,'height':1200}


def fixture():
    """Level-2 sample with a rotated (rigid) badge group, a rich-text headline and a button."""
    s=json.loads((ROOT/'input/chameleon_sample_level2_core.json').read_text());s=s.get('source',s)
    for e in s['elements']:
        if e['id'] in ('badge_bg','badge_label'):e['rotation']=-12
    return s


def designer_target(source,A):
    """A valid target v1 cannot express: the headline keeps a smaller font than v1's refit."""
    plan={'groups':[{'id':'logo','x':.02,'y':.1,'w':.08,'h':.3},{'id':'product','x':.7,'y':.05,'w':.25,'h':.9},
                    {'id':'headline','x':.12,'y':.1,'w':.5,'h':.4},{'id':'subhead','x':.12,'y':.55,'w':.5,'h':.15},
                    {'id':'cta','x':.12,'y':.75,'w':.2,'h':.18},{'id':'badge','x':.55,'y':.02,'w':.14,'h':.3}]}
    d=apply_plan(source,TARGET,A,plan)
    h=next(e for e in d['elements'] if e['id']=='headline');h['font_size']=max(8,int(h['font_size']*.8))
    for e in d['elements']:
        if e['id'].startswith('deco'):  # keep decorations on the wide canvas
            e['x']=min(max(0,e['x']),TARGET['width']-e['width']);e['y']=min(max(0,e['y']),TARGET['height']-e['height'])
    return d


class PlanContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=fixture();cls.A=analyze(cls.source);cls.target=designer_target(cls.source,cls.A)

    def test_materialization_fixture_rich_text_and_rigid_group(self):
        s,A,t=self.source,self.A,self.target
        self.assertIn('RigidBlock',{type(b).__name__ for b in A.all_blocks()})
        self.assertTrue(passes(validate(s,t)))
        v1=materialize_plan(s,TARGET,A,design_to_plan(t,A))
        # v1 cannot express this target (refit headline font, moved decoration): rejected or tagged lossy, never an SFT target.
        self.assertTrue(v1.status!='ok' or plan_fidelity(t,v1.design)['class']=='lossy')
        v2=materialize_plan(s,TARGET,A,design_to_plan_v2(t,s,A))
        self.assertEqual(v2.status,'ok');self.assertTrue(v2.raw_valid);self.assertEqual(plan_fidelity(t,v2.design)['class'],'exact')
        src={e['id']:e for e in s['elements']}
        for e in v2.design['elements']:
            for k,v in src[e['id']].items():
                if k not in ('x','y','width','height','font_size'):self.assertEqual(e[k],v,(e['id'],k))
        bad=design_to_plan_v2(t,s,A)
        next(p for p in bad['elements'] if p['id']=='badge_label')['x']+=.3
        self.assertIn('group_box_inconsistent',validate_plan(A,bad,target=TARGET,source=s).codes)

    def test_malformed_plans_rejected_before_repair(self):
        base=design_to_plan(self.target,self.A)
        def g(plan,i):return plan['groups'][i]
        cases={
            'missing_block':lambda p:p['groups'].pop(),
            'duplicate_id':lambda p:p['groups'].append(dict(g(p,0))),
            'unknown_id':lambda p:g(p,0).update(id='nope'),
            'id_not_string':lambda p:g(p,0).update(id=3),
            'boolean_value':lambda p:g(p,0).update(x=True),
            'non_numeric':lambda p:g(p,0).update(w='0.2'),
            'non_finite':lambda p:g(p,0).update(h=math.nan),
            'negative_origin':lambda p:g(p,0).update(x=-.01),
            'non_positive_size':lambda p:g(p,0).update(w=0),
            'out_of_bounds':lambda p:g(p,0).update(x=.9,w=.2),
            'unexpected_field':lambda p:g(p,0).update(font='x'),
            'unexpected_top_field':lambda p:p.update(code='rm -rf'),
            'bad_arrangement':lambda p:p.update(arrangement='collage'),
            'groups_not_list':lambda p:p.update(groups={}),
        }
        for code,mutate in cases.items():
            with self.subTest(code=code):
                plan=copy.deepcopy(base);mutate(plan)
                self.assertIn(code,validate_plan(self.A,plan).codes)
                with patch('engine.repair.contain') as contain:
                    with self.assertRaises(PlanRejected):apply_plan(self.source,TARGET,self.A,plan)
                    contain.assert_not_called()
                self.assertEqual(materialize_plan(self.source,TARGET,self.A,plan).status,'rejected')
        locks={'headline':{'x':0,'y':0,'width':100,'height':50}}
        self.assertIn('lock_conflict',validate_plan(self.A,base,locks=locks,target=TARGET).codes)
        self.assertIn('partial_lock_v1',validate_plan(self.A,base,locks={'cta_bg':{'x':0,'y':0}},target=TARGET).codes)
        for text in ('','not json',"{'groups': []}",'[1,2]','\x00garbage{'):
            self.assertIsNone(parse_plan_text(text))

    def test_edit_action_orientation_and_rejection(self):
        s,A=self.source,self.A
        state=apply_plan(s,TARGET,A,design_to_plan(self.target,A))  # a weaker state; validity is checked on the result
        rev=DesignRevision.create(state);action=edit_action(state,self.target,'imitate')
        result=apply_edit(s,rev,action_to_proposal(action,rev,provenance='deterministic_synthetic'))
        self.assertEqual(plan_fidelity(self.target,result)['class'],'exact')
        for bad in ({'headline':{'content':'NEW'}},{'headline':{'x':True}},{'ghost':{'x':1}},{'headline':{'width':-4}}):
            with self.subTest(patch=bad):
                with self.assertRaises(PlanRejected):action_to_proposal({**action,'patch':bad},rev)
        with self.assertRaises(PlanRejected):action_to_proposal(action,rev,locks={next(iter(action['patch'])):{'x':0}})


class DatasetContractTests(unittest.TestCase):
    def test_orientation_provenance_split_and_idempotent_export(self):
        from training import edit_dataset as ed
        s=fixture();A=analyze(s);final=designer_target(s,A)
        state=apply_plan(s,TARGET,A,design_to_plan(final,A))
        demo=lambda fam,src:{'demo_id':ed.sha256_text(fam)[:16],'design_id':fam,'family':fam,'source':src,'target':TARGET,
                             'provenance':'deterministic_synthetic','teacher':'fixture','code_revision':{},
                             'final':{'design':final,'template':'scale','overall':90.0,'defects':{}},
                             'states':[{'design':state,'template':'scale','overall':70.0,'defects':{'overflow':['headline']},'relation':'weaker_candidate'}]}
        level3=json.loads((ROOT/'input/chameleon_sample_level3_advanced.json').read_text())['source']
        manifest={'schema_version':1,'families':{'fix-train':'train','fix-test':'test','suite-copy':'train'},'aliases':{},'excluded_families':[]}
        with tempfile.TemporaryDirectory() as tmp:
            demos=Path(tmp)/'demos.jsonl'
            demos.write_text(''.join(json.dumps(d)+'\n' for d in (demo('fix-train',s),demo('fix-test',s),demo('suite-copy',level3))))
            outs=[]
            for n in range(2):
                out=Path(tmp)/f'data{n}';m=ed.build([demos],out,manifest,render=False)
                outs.append({f:(out/f).read_bytes() for f in ('plan_sft.jsonl','edit_sft.jsonl','manifest.json')})
            self.assertEqual(outs[0],outs[1])  # idempotent
            self.assertEqual(m['counts']['skipped_test'],1);self.assertEqual(m['counts']['excluded_suite_source_hash'],1)
            report=ed.validate_dataset(Path(tmp)/'data0');self.assertTrue(report['ok'],report)
            edit=ed.load_rows(Path(tmp)/'data0','edit')[0]
            self.assertEqual((edit['orientation'],edit['provenance'],edit['split']),('base_to_result','deterministic_synthetic','train'))
            self.assertEqual(edit['base_revision'],design_hash(state))
            flipped={**edit,'task':json.dumps({**json.loads(edit['task']),'state':final})}
            self.assertIn('base_revision_mismatch',ed.validate_row(flipped))
            plan=ed.load_rows(Path(tmp)/'data0','plan')[0];self.assertEqual(plan['plan_schema'],PLAN_V2)
            for row,code in (({**plan,'provenance':'explicit_human'},'human_without_review_record'),({**plan,'provenance':'human'},'unknown_provenance'),
                             ({**plan,'split':'test'},'non_exported_split')):
                self.assertIn(code,ed.validate_row(row))


class RuntimeAdapterTests(unittest.TestCase):
    def test_incompatible_bundles_fail_closed(self):
        from engine import vlm
        from engine.neural_proposals import propose
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(vlm.check_bundle(tmp)['status'],'missing_bundle')
            mf=Path(tmp)/'bundle_manifest.json'
            good={'bundle_schema':vlm.BUNDLE_SCHEMA,'adapter_id':'tiny-qwen2.5-vl-test','plan_schema':PLAN_V2,
                  'preprocessing':{'image_transform':'letterbox_white_672','image_size':672},'files':{'adapter/x.bin':'0'}}
            for patch_,status in (({'bundle_schema':'other'},'wrong_bundle_schema'),({'adapter_id':'gpt-x'},'unknown_adapter'),
                                  ({'preprocessing':{'image_transform':'resize_448','image_size':448}},'incompatible_transform'),
                                  ({'plan_schema':'layout-plan/9'},'incompatible_schema'),({},'missing_file')):
                with self.subTest(status=status):
                    mf.write_text(json.dumps({**good,**patch_}));self.assertEqual(vlm.check_bundle(tmp)['status'],status)
            env={'CHAMELEON_VLM':'1','CHAMELEON_VLM_BACKEND':'transformers','CHAMELEON_VLM_BUNDLE':tmp}
            with patch.dict(os.environ,env):
                s=fixture();design,event=propose(s,TARGET,analyze(s))
            self.assertIsNone(design);self.assertEqual(event['status'],'weights_missing')

    def test_processor_collator_masking_tiny_model(self):
        try:
            import torch
            from peft import get_peft_model
            from training.vlm_adapters import collate,encode_example,assert_supervision,gradient_report,load_base,lora_config,supervised_text
            from engine.vlm import adapter
            a=adapter('tiny-qwen2.5-vl-test');model,processor=load_base(a,None,'cpu','float32')
        except Exception as exc:
            self.skipTest('tiny test model not cached locally: '+type(exc).__name__)
        from PIL import Image
        completion=json.dumps({'schema':PLAN_V2,'groups':[],'elements':[]})
        ex=encode_example(processor,[Image.new('RGB',(112,112),'white')],'Return the plan.',completion)
        self.assertTrue(assert_supervision(processor,ex));self.assertTrue(supervised_text(processor,ex).startswith(completion))
        self.assertLess(int((ex['labels']!=-100).sum()),int(ex['input_ids'].shape[0])//2)
        torch.manual_seed(0);model=get_peft_model(model,lora_config(a,8))
        loss=model(**{k:v for k,v in collate([ex],processor.tokenizer.pad_token_id or 0).items()}).loss
        self.assertTrue(torch.isfinite(loss));loss.backward()
        report=gradient_report(model)
        self.assertGreater(report['with_grad'],0);self.assertEqual(report['grad_non_lora'],[]);self.assertEqual(report['trainable_non_lora'],[])


if __name__=='__main__':unittest.main()
