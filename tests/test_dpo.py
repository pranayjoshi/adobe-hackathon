"""Lean checks for guide 16: preference-pair contract, DPO reference/gradients on the tiny model, export -> runtime."""
import copy,json,math,tempfile,unittest
from pathlib import Path
from engine.analysis import analyze
from engine.contracts import design_hash
from engine.neural_proposals import PLAN_V2,design_to_plan,design_to_plan_v2,materialize_plan,parse_plan_text
from test_vlm_training import ROOT,TARGET,designer_target,fixture

MANIFEST={'schema_version':1,'families':{'fix-train':'train','fix-val':'validation'},'aliases':{},'excluded_families':[]}
REVIEW={'reviewer_id':'r1','committed_at':'2026-10-25T00:00:00Z'}


class PreferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training import edit_dpo as ed
        cls.ed=ed;cls.s=fixture();cls.A=analyze(cls.s)
        cls.final=designer_target(cls.s,cls.A);cls.state=copy.deepcopy(cls.final)
        for e in cls.state['elements']:
            if e['id']=='headline':e['x']+=60  # a second valid layout of the same task
        cls.ctx={'source':cls.s,'target':TARGET,'plan_schema':PLAN_V2}
        # orientation-independent fixture: which side the surrogate prefers is measured, not assumed
        r={k:ed.surrogate_reward(cls.s,TARGET,ed.materialize_side('plan',cls.ctx,ed.side_from_design('plan',cls.ctx,d,cls.A),cls.A)[1],cls.A)
           for k,d in (('x',cls.final),('y',cls.state))}
        assert r['x']!=r['y'],r
        cls.lo,cls.hi=sorted(r,key=r.get)

    def rec(self,pid,verdict,presentation,provenance='deterministic_synthetic',family='fix-train',**over):
        r={'pref_schema':self.ed.PREF_SCHEMA,'pref_id':pid,'kind':'plan','provenance':provenance,'source_family':family,'design_id':family,
           'verdict':verdict,'presentation':presentation,'context':self.ctx,'candidates':{'x':{'design':self.final},'y':{'design':self.state}}}
        if provenance=='explicit_human':r['review']={**REVIEW,'verdict':verdict}
        else:r['teacher']='fixture'
        r.update(over);return r

    def build(self,tmp,records):
        p=Path(tmp)/'prefs.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in records))
        m=self.ed.build([p],Path(tmp)/'data',MANIFEST,render=False)
        rows=lambda v:[r for k in ('plan',) for s in ('train','validation') for r in self.ed.load_pairs(Path(tmp)/'data'/v,k,s)]
        return m,rows

    def test_identical_context_orientation_nonbinary_and_human_disagreement(self):
        lo,hi=self.lo,self.hi
        bad_plan=json.dumps({'schema':PLAN_V2,'groups':[{'id':'ghost','x':0,'y':0,'w':.1,'h':.1}],'elements':[]})
        records=[
            # human prefers the side the surrogate scores LOWER, displayed on B: kept + flagged, never dropped
            self.rec('p1','B',{'A':hi,'B':lo},'explicit_human'),
            # synthetic agreeing with the surrogate, chosen shown on B after randomization
            self.rec('p2','B',{'A':lo,'B':hi},family='fix-val'),
            self.rec('p2-dup','B',{'A':lo,'B':hi},family='fix-val'),                      # exact duplicate
            self.rec('p3','tie',{'A':'x','B':'y'},'explicit_human'),
            self.rec('p4','both_bad',{'A':'y','B':'x'},'explicit_human'),
            self.rec('p5','unsure',{'A':'x','B':'y'},'explicit_human'),
            self.rec('p6','A',{'A':'x','B':'y'},candidates={'x':{'design':self.final,'target':{'width':10,'height':10}},'y':{'design':self.state}}),
            self.rec('p7','A',{'A':'x','B':'y'},candidates={'x':{'completion':bad_plan},'y':{'design':self.state}}),
            {**self.rec('p8','A',{'A':'x','B':'y'},'explicit_human'),'review':None},
        ]
        level3=json.loads((ROOT/'input/chameleon_sample_level3_advanced.json').read_text())['source']
        records.append(self.rec('p9','A',{'A':'x','B':'y'},context={'source':level3,'target':TARGET}))
        with tempfile.TemporaryDirectory() as tmp:
            m,rows=self.build(tmp,records);c=m['counts']
            binary=rows('human_plus_synthetic')
            self.assertEqual(sorted(r['pref_id'] for r in binary),['p1','p2'])
            self.assertEqual([r['pref_id'] for r in rows('human_only')],['p1'])
            self.assertEqual([r['pref_id'] for r in rows('synthetic_only')],['p2'])
            self.assertEqual(m['nonbinary'],{'tie':1,'both_bad':1,'unsure':1})
            self.assertEqual(c['conflicting_pairs'],1)  # p1 vs p2: opposite orientation on the same context is kept
            for why in ('duplicate_pair','context_mismatch','invalid_side','human_without_review_record'):self.assertEqual(c['excluded_'+why],1,why)
            self.assertTrue(any(k.startswith('excluded_visual_suite') for k in c))
            by={r['pref_id']:r for r in binary};designs={'x':self.final,'y':self.state}
            for r in binary:
                self.assertEqual(r['presentation'][r['verdict']],r['chosen_id'])
                self.assertEqual(len({r['source_hash'],design_hash(self.s)}),1)
                o=materialize_plan(self.s,TARGET,self.A,parse_plan_text(r['chosen']))   # runtime parse path
                self.assertEqual(o.status,'ok');self.assertEqual(design_hash(o.design),r['chosen_revision'])
                self.assertEqual(sorted(e['id'] for e in o.design['elements']),sorted(e['id'] for e in self.s['elements']))  # exact coverage
                self.assertLess(max(abs(a[k]-b[k]) for a,b in zip(sorted(o.design['elements'],key=lambda e:e['id']),sorted(designs[r['chosen_id']]['elements'],key=lambda e:e['id'])) for k in ('x','y','width','height')),1.0)
            self.assertEqual((by['p1']['chosen_id'],by['p1']['reward_disagreement']),(lo,True))
            self.assertEqual((by['p2']['chosen_id'],by['p2']['reward_disagreement']),(hi,False))
            self.assertEqual(m['audit']['by_provenance'],{'explicit_human:reward_disagreement':1,'deterministic_synthetic:agrees':1})
            rep=self.ed.validate_dataset(Path(tmp)/'data'/'human_plus_synthetic',audit=True);self.assertTrue(rep['ok'],rep)
            flipped={**by['p2'],'verdict':'A'}
            self.assertIn('orientation_mismatch',self.ed.validate_pair_row(flipped))
        with tempfile.TemporaryDirectory() as tmp:  # the same disagreement from a synthetic source is still an ordering error
            m,rows=self.build(tmp,[self.rec('q1','A',{'A':lo,'B':hi})])
            self.assertEqual(m['counts']['excluded_synthetic_misordered'],1);self.assertEqual(rows('synthetic_only'),[])

    def test_train_vlm_check_audits_human_disagreement(self):
        from training.reward import reward
        from training.train_vlm import check
        plans={k:design_to_plan_v2(d,self.s,self.A) for k,d in (('x',self.final),('y',self.state))}
        sc={k:reward(self.s,TARGET,p,self.A) for k,p in plans.items()}
        lo,hi=sorted(sc,key=sc.get)
        row={'task':json.dumps({'source':self.s,'target':TARGET}),'images':[],'chosen':json.dumps(plans[lo]),'rejected':json.dumps(plans[hi])}
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'dpo.jsonl'
            f.write_text(json.dumps({**row,'provenance':'explicit_human','verdict':'A','review':{**REVIEW,'verdict':'A'}})+'\n')
            r=check(tmp,'dpo');self.assertEqual((r['misordered_pairs'],r['reward_disagreements'],r['invalid_plans']),(0,1,0));self.assertTrue(r['ok'],r)
            f.write_text(json.dumps(row)+'\n')  # legacy rubric row (no provenance): ordering still enforced
            r=check(tmp,'dpo');self.assertEqual(r['misordered_pairs'],1);self.assertFalse(r['ok'])


class TinyModelDPOTests(unittest.TestCase):
    def test_reference_fixed_gradients_on_policy_lora_and_runtime_export(self):
        try:
            import torch
            from peft import get_peft_model
            from engine.vlm import adapter
            from training.vlm_adapters import load_base,lora_config
            a=adapter('tiny-qwen2.5-vl-test');model,processor=load_base(a,None,'cpu','float32')
        except Exception as exc:
            self.skipTest('tiny test model not cached locally: '+type(exc).__name__)
        from PIL import Image
        from training import edit_dpo as ed
        from training.vlm_adapters import encode_example,assert_supervision,export_bundle,gradient_report
        torch.manual_seed(0);model=get_peft_model(model,lora_config(a,4))
        with torch.no_grad():  # stand-in SFT adapter: non-zero LoRA-B so the SFT policy differs from the base model
            for n,p in model.named_parameters():
                if 'lora_B' in n:p.normal_(0,.02)
        ed.attach_reference(model);model.eval()
        img=[Image.new('RGB',(112,112),'white')]
        c=encode_example(processor,img,'Return the plan.',json.dumps({'schema':PLAN_V2,'groups':[]}));j=encode_example(processor,img,'Return the plan.',json.dumps({'schema':PLAN_V2,'groups':[],'elements':[]}))
        assert_supervision(processor,c);assert_supervision(processor,j)
        def ref():
            model.set_adapter('ref')
            with torch.no_grad():out=(float(ed.sequence_logp(model,c,'cpu','float32')),float(ed.sequence_logp(model,j,'cpu','float32')))
            model.set_adapter('default')
            for n,p in model.named_parameters():
                if '.ref.' in n:p.requires_grad_(False)
            return out
        r0=ref();fp=ed.reference_fingerprint(model)
        with torch.no_grad(),model.disable_adapter():base=float(ed.sequence_logp(model,c,'cpu','float32'))
        self.assertNotEqual(base,r0[0])  # reference is the SFT adapter copy, not the disabled-adapter base
        pc,pr=ed.sequence_logp(model,c,'cpu','float32'),ed.sequence_logp(model,j,'cpu','float32')
        loss,_=ed.dpo_loss(pc,pr,r0[0],r0[1],.1)
        self.assertTrue(torch.isfinite(loss));self.assertAlmostEqual(float(loss),math.log(2),places=6)  # policy == reference at init
        loss.backward();g=gradient_report(model)
        self.assertGreater(g['with_grad'],0);self.assertEqual(g['grad_non_lora'],[])
        self.assertEqual([n for n,p in model.named_parameters() if p.grad is not None and '.default.' not in n],[])
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1.0).step()
        self.assertEqual(ref(),r0);self.assertEqual(ed.reference_fingerprint(model),fp)
        with torch.no_grad():self.assertNotEqual(float(ed.sequence_logp(model,c,'cpu','float32')),r0[0])
        model.delete_adapter('ref')
        from engine.vlm import TransformersPolicy,check_bundle
        with tempfile.TemporaryDirectory() as tmp:
            export_bundle(Path(tmp)/'b',model,processor,a,PLAN_V2,tasks=['plan'],train_config={'stage':'dpo'},status='trained_dpo_smoke')
            self.assertEqual(check_bundle(Path(tmp)/'b',verify_hashes=True)['status'],'ok')
            self.assertFalse((Path(tmp)/'b'/'adapter'/'ref').exists())
            s=fixture();A=analyze(s)
            from engine.neural_proposals import plan_prompt_v2
            text,_=TransformersPolicy.load(Path(tmp)/'b','cpu').generate([Image.new('RGB',(112,112),'white')],plan_prompt_v2(s,TARGET,A),8)
            plan=parse_plan_text(text)
            if plan is not None:self.assertIn(materialize_plan(s,TARGET,A,plan).status,('ok','rejected','materialization_failed','invalid_design'))


if __name__=='__main__':unittest.main()
