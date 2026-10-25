"""Lean guide17 contracts: preservation reward, bounded replay, conditional training."""
import copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from engine.analysis import analyze
from engine.neural_proposals import PLAN_V2,design_to_plan_v2,materialize_plan
from training.grpo_training import GRPOConfig,group_advantages,kernel_smoke,prerequisites,rollout
from training.reward import reward_record


def fixture():
    return {'canvas':{'width':200,'height':100},'elements':[{'id':'shape','type':'shape','shape_type':'rectangle','fill_color':'#ac2266','x':20,'y':20,'width':40,'height':40}]}


class GRPOContracts(unittest.TestCase):
    def test_reward_rejects_content_drop_asset_injection_and_lock_hiding(self):
        source=fixture();A=analyze(source);plan=design_to_plan_v2(source,source,A)
        good=reward_record(source,source['canvas'],plan,A);self.assertTrue(good['raw_valid']);self.assertTrue(good['repaired_valid']);self.assertGreaterEqual(good['reward'],0)
        for mutate in ('drop','asset','hide','locked'):
            bad=copy.deepcopy(plan);locks=None
            if mutate=='drop':bad['elements']=[]
            if mutate=='asset':bad['elements'][0]['asset_id']='invented.png'
            if mutate=='hide':bad['elements'][0]['w']=0
            if mutate=='locked':bad['elements'][0]['x']+=.2;locks={'shape':{'x':20,'y':20,'width':40,'height':40}}
            r=reward_record(source,source['canvas'],bad,A,locks=locks,preference=1000000)
            self.assertEqual(-1,r['reward'],mutate)
        # An invalid raw design cannot gain reward from a repair/containment pass.
        invalid=copy.deepcopy(plan);invalid['elements'][0]['x']=.9
        r=reward_record(source,source['canvas'],invalid,A,repair=True)
        self.assertEqual(-1,r['reward'])

    def test_equal_rewards_finite_loss_and_replay_stops(self):
        smoke=kernel_smoke();self.assertTrue(smoke['finite_gradients']);self.assertTrue(smoke['nonzero_gradient']);self.assertEqual(0,smoke['equal_reward_loss'])
        self.assertEqual([0.,0.,0.,0.],group_advantages([1.]*4))
        source=fixture();plan=design_to_plan_v2(source,source,analyze(source));completions=[json.dumps(plan)]*4
        result=rollout(source,source['canvas'],completions);self.assertTrue(result['zero_reward_variance']);self.assertEqual(4,len(result['records']));self.assertTrue(result['records'][1]['cache_hit'])
        cancelled=rollout(source,source['canvas'],completions,cancelled=lambda:True);self.assertEqual('cancelled',cancelled['status']);self.assertEqual([],cancelled['records'])
        deadline=rollout(source,source['canvas'],completions,deadline=time.monotonic()-1);self.assertEqual('deadline',deadline['status'])

    def test_conditional_run_has_no_fabricated_training_or_human_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=prerequisites(Path(tmp))
        self.assertFalse(report['ready']);self.assertFalse(report['training_performed']);self.assertFalse(report['quality_improvement_claim']);self.assertGreater(len(report['pending']),4)

    def test_edit_helper_disabled_and_cancelled_worker_are_typed(self):
        from engine.neural_proposals import propose_edit,_infer_request
        from engine.contracts import DesignRevision
        source=fixture();revision=DesignRevision.create(source)
        with patch('engine.vlm.status',return_value={'enabled':False}):
            proposal,event=propose_edit(source,source['canvas'],analyze(source),revision,{})
        self.assertIsNone(proposal);self.assertEqual('disabled',event['status']);self.assertEqual(0,event['model_calls'])
        plan,event=_infer_request({'task':'edit','source':source,'state':source,'schema':'edit-action/1','prompt':'fixture'},timeout=.001,cancelled=lambda:True)
        self.assertIsNone(plan);self.assertEqual('cancelled',event['status']);self.assertEqual(0,event['model_calls'])
