"""C1 semantic structure: graph sidecar, group policies and hierarchical blocks.

Behaviour-level checks only (plan 20 C1 / guides 01-02 lean tests).
"""
import copy
import os
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from engine.analysis import analyze
from engine.group_policies import enumerate_configs, validate
from engine.pipeline import generate
from engine.relations import GRAPH_VERSION

PHOTO = 'crello/04e19d5b19f51534a0dd/0.png'
FLAT = 'crello/0e9b02cf2ed6d45d7e71/0.png'
LOGO = 'solstice_logo_lockup.png'
FAST = {'render': False, 'solver': False, 'retrieval': False, 'refine': False}
STYLE = ('content', 'font_family', 'color', 'style_runs', 'group_id', 'z_index', 'opacity', 'rotation', 'asset_id', 'text_align', 'letter_spacing')


def text(i, x, y, fs, content='Some text', w=None, h=None, z=5, **kw):
    return dict(id=i, type='text', x=x, y=y, width=w or fs * len(content) * .6, height=h or fs * 1.2, font_family='Solstice Sans',
                font_size=fs, content=content, color='rgba(255,255,255,1)', z_index=z, **kw)


def image(i, x, y, w, h, asset, z=1):
    return dict(id=i, type='image', x=x, y=y, width=w, height=h, asset_id=asset, fit='fill', z_index=z)


def shape(i, x, y, w, h, z=3, **kw):
    return dict(id=i, type='shape', shape_type='rectangle', fill_color='rgba(232,106,58,1)', x=x, y=y, width=w, height=h, z_index=z, **kw)


def offer():
    """Display offer: '40' with a nested '%' and a tight 'OFF' line, copy, CTA, a
    logo with its wordmark and an explicit badge group."""
    els = [
        text('big', 100, 300, 300, '40', w=420, h=300),
        text('pct', 400, 380, 60, '%', w=60, h=60, z=6),
        text('off', 100, 590, 200, 'OFF', w=420, h=200),
        text('copy', 100, 830, 40, 'Annual membership offer', w=700, h=50),
        image('logo', 100, 60, 320, 100, LOGO, z=4),
        text('mark', 160, 135, 40, 'BRAND', w=200, h=48, text_align='center'),
        shape('b_bg', 100, 920, 240, 50, group_id='badge'),
        text('b_lbl', 100, 932, 20, 'LIMITED', w=240, h=26, group_id='badge'),
        shape('cta_bg', 600, 1100, 300, 80),
        text('cta_lbl', 600, 1120, 28, 'Join now', w=300, h=40),
    ]
    return {'canvas': {'width': 1080, 'height': 1350}, 'elements': els}


def hgen(*a, **k):
    """generate() under the hierarchical switch (pipeline reads the env var)."""
    with patch.dict(os.environ, {'CHAMELEON_GROUP_POLICIES': 'hierarchical'}):
        return generate(*a, **k)


def owned_leaves(A):
    by = {p.group_id: p for p in A.policies}
    return [i for p in A.policies for i in p.owned_ids(by)]


class SemanticStructureTests(unittest.TestCase):
    def test_explicit_intent_changes_real_blocks_and_survives_conflicting_model(self):
        from engine.blocks import StrictRigidBlock
        from engine.learning import apply_roles,load_role_artifact,role_mode
        s=offer();before=copy.deepcopy(s);base=analyze(s,group_policies='hierarchical')
        stack=next(policy for policy in base.policies if policy.kind=='display_stack')
        A=analyze(s,group_policies='hierarchical',overrides={'roles':{'copy':'headline'},'group_modes':{stack.group_id:'rigid'}})
        self.assertIsInstance(next(block for block in A.blocks if block.id==stack.group_id),StrictRigidBlock)
        self.assertEqual(next(policy for policy in A.policies if policy.group_id==stack.group_id).authority,'explicit')
        old_graph=A.graph;model,status=load_role_artifact();self.assertEqual(status,'ok')
        conflict=copy.deepcopy(model);conflict['calibration']['thresholds']['body']=.9
        p=[float(role=='body') for role in model['classes']]
        with patch('engine.learning.load_role_artifact',return_value=(conflict,'ok')),patch('engine.learning.predict_proba',return_value=p):
            apply_roles(s,A,enabled='calibrated')
        self.assertEqual(A.roles['copy'],'headline');self.assertEqual(A.role_sources['copy'],'explicit')
        self.assertEqual(A.role_confidence['copy'],1);self.assertIsNot(A.graph,old_graph)
        self.assertEqual(A.graph.nodes['copy']['role'],'headline');self.assertEqual(s,before)
        with patch.dict(os.environ,{'CHAMELEON_ROLE_MODEL':'calibrated'}):
            self.assertEqual(role_mode('legacy'),'legacy');self.assertEqual(role_mode(False),'legacy')
        with self.assertRaises(ValueError):analyze(s,overrides={'roles':{'copy':'hero'}})
        with self.assertRaises(ValueError):analyze(s,group_policies='hierarchical',overrides={'group_modes':{'missing':'rigid'}})
        rigid=next(policy for policy in base.policies if policy.kind=='logo_lockup')
        with self.assertRaises(ValueError):analyze(s,group_policies='hierarchical',overrides={'group_modes':{rigid.group_id:'support'}})

    def test_calibrated_artifact_rejects_corrupt_tree_and_shadow_preserves_roles(self):
        from engine import learning
        s=offer();good,status=learning.load_role_artifact();self.assertEqual(status,'ok')
        corrupt=copy.deepcopy(good);corrupt['trees'][0]['left'][0]=0
        corrupt['artifact_hash']=learning._sha({k:v for k,v in corrupt.items() if k not in ('training','artifact_hash')})[:16]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'roles.json';path.write_text(json.dumps(corrupt))
            self.assertEqual(learning.load_role_artifact(path)[1],'invalid tree structure')
            with patch.dict(os.environ,{'CHAMELEON_ROLE_WEIGHTS_V2':str(path)}):
                A=analyze(s);initial=copy.deepcopy(A.roles);result=learning.apply_roles(s,A,enabled='calibrated')
            self.assertEqual(result['status'],'fallback_heuristic');self.assertEqual(A.roles,initial)
        A=analyze(s);initial=copy.deepcopy(A.roles)
        self.assertEqual(learning.apply_roles(s,A,enabled='shadow')['status'],'shadow')
        self.assertEqual(A.roles,initial)

    def test_opt_in_card_grid_owns_illustration_and_preserves_support(self):
        s={'canvas':{'width':300,'height':200},'elements':[
            shape('left',0,0,150,200,z=0),shape('right',150,0,150,200,z=0),
            text('a',10,20,18,'Hello',w=130,h=30),text('b',160,20,18,'World',w=130,h=30),
            image('illustration',20,80,50,50,LOGO,z=4)]}
        baseline=analyze(s,group_policies='hierarchical')
        self.assertFalse(any(policy.kind=='card_grid' for policy in baseline.policies))
        A=analyze(s,group_policies='hierarchical',features='panels,card_images,tiles')
        self.assertEqual(A.policy_status['status'],'ok')
        grid=next(policy for policy in A.policies if policy.kind=='card_grid')
        self.assertIn('illustration',grid.member_ids)
        leaves=owned_leaves(A);self.assertEqual(len(leaves),len(set(leaves)))
        validate(A.policies,[element['id'] for element in s['elements']])
    def test_reflow_with_locked_child_and_ambiguous_overlap_fallback(self):
        s = offer()
        A = analyze(s, group_policies='hierarchical')
        stack = next(p for p in A.policies if p.kind == 'display_stack')
        lock = next(p for p in A.policies if p.kind == 'text_lockup')
        self.assertEqual(set(lock.member_ids), {'big', 'pct'})
        self.assertEqual(lock.parent, stack.group_id)
        self.assertEqual(set(stack.member_ids), {'big', 'pct', 'off'})
        target = {'width': 1584, 'height': 396}
        configs = enumerate_configs(stack, s, A, target)
        self.assertTrue(0 < len(configs) <= 8)
        self.assertEqual({c.penalties['axis_change'] for c in configs}, {0.0, 1.0})  # vertical and side-by-side
        for c in configs:  # nested lockup keeps its relative geometry in every arrangement
            dx = c.child_patches['pct']['x'] - c.child_patches['big']['x']
            self.assertAlmostEqual(dx / c.child_patches['big']['width'], 300 / 420, places=3)
        # A locked child either stays exact or the configuration is infeasible.
        pin = {'pct': {'x': 700.0, 'y': 50.0, 'width': 30.0, 'height': 30.0}}
        locked = enumerate_configs(lock, s, A, target, locks=pin)
        self.assertTrue(locked)
        for c in locked:
            if c.feasible:
                p = c.place(0, 0)['pct']
                for k, v in pin['pct'].items():
                    self.assertAlmostEqual(p[k], v, delta=.6)
        self.assertTrue(any(c.feasible for c in locked))
        bad = enumerate_configs(lock, s, A, target, locks={'pct': {'x': 1, 'y': 1, 'width': 30.0, 'height': 90.0}})
        self.assertTrue(bad and not any(c.feasible for c in bad))
        # The full pipeline honours the lock exactly under hierarchical grouping.
        out = hgen(s, target, top_k=1, locks=pin, options=FAST)
        self.assertTrue(all(ch['passed'] for ch in out[0]['checks'] if ch['name'].startswith('Locked')))
        # Equal-size overlapping copy is ambiguous: it stays two independent units.
        amb = {'canvas': {'width': 600, 'height': 800}, 'elements': [text('a', 20, 20, 20, w=100, h=40), text('b', 20, 30, 20, w=100, h=40)]}
        self.assertEqual(len(analyze(amb, group_policies='hierarchical').blocks), 2)

    def test_mixed_hierarchy_preserves_leaves_styles_and_rigid_logo(self):
        s = offer()
        before = copy.deepcopy(s)
        A = analyze(s, group_policies='hierarchical')
        validate(A.policies, [e['id'] for e in s['elements']])
        leaves = owned_leaves(A)
        self.assertEqual(len(leaves), len(set(leaves)))  # single layout owner per leaf
        block_leaves = [i for b in A.all_blocks() for i in b.ids]
        self.assertEqual(sorted(block_leaves), sorted(set(block_leaves)))
        kinds = {p.kind: p for p in A.policies}
        self.assertEqual(kinds['logo_lockup'].mode, 'rigid')
        self.assertEqual(set(kinds['logo_lockup'].member_ids), {'logo', 'mark'})
        self.assertEqual(next(p for p in A.policies if 'b_bg' in p.member_ids).authority, 'explicit')
        self.assertIn('button', kinds)
        for target in ({'width': 728, 'height': 90}, {'width': 160, 'height': 600}):
            d = hgen(s, target, top_k=1, options=FAST)[0]['design']
            by = {e['id']: e for e in d['elements']}
            self.assertEqual(sorted(by), sorted(e['id'] for e in s['elements']))
            for e in s['elements']:
                self.assertEqual({k: e.get(k) for k in STYLE}, {k: by[e['id']].get(k) for k in STYLE})
            lo, mk = by['logo'], by['mark']
            k = lo['width'] / 320
            self.assertAlmostEqual(lo['height'] / 100, k, places=2)
            self.assertAlmostEqual((mk['x'] + mk['width'] / 2 - lo['x']) / k, 260 - 100, delta=3)
        self.assertEqual(s, before)

    def test_photo_is_never_a_flexible_support_and_no_leaf_twice(self):
        els = [image('photo', 0, 100, 600, 500, PHOTO, z=1), image('panel', 650, 100, 400, 500, FLAT, z=1),
               text('on_photo', 60, 200, 40, 'Over the photo', z=5), text('on_panel', 700, 200, 30, 'On the card', z=5),
               text('note', 700, 300, 20, 'More detail here', z=5)]
        s = {'canvas': {'width': 1080, 'height': 700}, 'elements': els}
        A = analyze(s, group_policies='hierarchical')
        cards = [b for b in A.blocks if getattr(b, 'kind', '') == 'card']
        self.assertEqual([b.panel['id'] for b in cards], ['panel'])
        self.assertEqual(set(cards[0].ids), {'panel', 'on_panel', 'note'})
        self.assertTrue(all(len(c.ids) == 1 for c in cards[0].children))  # paragraphs stay independent units inside the card
        self.assertNotIn('photo', [p.support_id for p in A.policies if p.mode == 'support'])
        ids = [i for b in A.all_blocks() for i in b.ids] + [e['id'] for e in A.background + A.decorations]
        self.assertEqual(sorted(ids), sorted(set(ids)))
        g = A.graph
        self.assertEqual(g.version, GRAPH_VERSION)
        self.assertIn(('photo', 'on_photo'), {(e['a'], e['b']) for e in g.edges_of('supports_on_photo')})
        self.assertNotIn('panel', {e['a'] for e in g.edges_of('supports_on_photo')})
        d = hgen(s, {'width': 1280, 'height': 720}, top_k=1, options=FAST)[0]['design']
        by = {e['id']: e for e in d['elements']}
        self.assertAlmostEqual(by['panel']['width'] / by['panel']['height'], 400 / 500, places=2)  # backplate never distorted

    def test_structure_is_robust_to_id_renames_and_independent_permutations(self):
        s = offer()
        ren = {e['id']: f'n{i:02d}' for i, e in enumerate(reversed(s['elements']))}
        t = copy.deepcopy(s)
        for e in t['elements']:
            e['id'] = ren[e['id']]
        # Permute elements whose paint order is equal (independent layers).
        t['elements'] = sorted(t['elements'], key=lambda e: (e['z_index'], e['id']), reverse=True)
        for mode in ('legacy', 'hierarchical'):
            A, B = analyze(s, group_policies=mode), analyze(t, group_policies=mode)
            self.assertEqual({ren[k]: v for k, v in A.roles.items()}, B.roles)
            shape = lambda X, f=lambda i: i: sorted((type(b).__name__, b.role, tuple(sorted(f(i) for i in b.ids))) for b in X.all_blocks())
            self.assertEqual(shape(A, ren.get), shape(B))
        self.assertEqual(sorted(p.kind for p in analyze(s, group_policies='hierarchical').policies),
                         sorted(p.kind for p in analyze(t, group_policies='hierarchical').policies))


if __name__ == '__main__':
    unittest.main()
