import copy
import unittest

from engine.contracts import Budget, DesignRevision, EditProposal, NoValidDesign, design_hash
from engine.controller import run_controller
from engine.materialize import apply_edit


def source():
    return {'canvas': {'width': 100, 'height': 100}, 'elements': [
        {'id': 'box', 'type': 'shape', 'shape_type': 'rectangle',
         'fill_color': 'rgba(0,0,0,1)', 'x': 0, 'y': 0, 'width': 10,
         'height': 10, 'z_index': 0}]}


def assess(design, operation):
    return {'valid': True, 'design': copy.deepcopy(design),
            'quality': {'defect_count': 0}, 'score': {'total': 50},
            'rubric': {'gate': True, 'tier2': {'defects': [], 'warnings': [],
                       'overall': 50 + design['elements'][0]['x']}}, 'trace': []}


def verify(design):
    return {'verified': True, 'status': 'verified', 'image': None}


class IterativeControllerTests(unittest.TestCase):
    def test_stale_immutable_and_locked_edits_cannot_change_source(self):
        original = source(); before = copy.deepcopy(original)
        revision = DesignRevision.create(original)
        proposals = [EditProposal('stale', 'move', {'box': {'x': 20}}),
                     EditProposal(revision.revision_hash, 'restyle', {'box': {'fill_color': 'red'}}),
                     EditProposal(revision.revision_hash, 'move', {'box': {'x': 20}}),
                     EditProposal(revision.revision_hash, 'invalid', {'box': {'width': float('nan')}})]
        for proposal in proposals:
            with self.subTest(proposal=proposal.operation), self.assertRaises(ValueError):
                apply_edit(original, revision, proposal, {'box': {'x': 0, 'y': 0}})
        self.assertEqual(original, before)

    def test_rejected_bad_edit_cycle_and_unavailable_judge_keep_verified_best(self):
        def expand(candidate, budget):
            bad = copy.deepcopy(candidate['design']); bad['elements'][0]['fill_color'] = 'red'
            yield 'bad higher reward', bad
            moved = copy.deepcopy(candidate['design']); moved['elements'][0]['x'] = 20
            yield 'move', moved
            yield 'cycle', copy.deepcopy(candidate['design'])
        artifacts = []
        result = run_controller(source(), [('seed', source())], assess=assess, expand=expand,
                                verify=verify, compare=lambda a, b: {'available': False},
                                mode='single', budget=Budget(seconds=5, max_edits=8),
                                artifact_callback=artifacts.append)[0]
        self.assertEqual(result['design']['elements'][0]['x'], 20)
        self.assertTrue(any(e['type'] == 'rejected' for e in result['controller']['events']))
        self.assertLessEqual(result['controller']['usage']['edits'], 8)
        self.assertTrue(all(e['assessment']['valid'] for e in artifacts))

    def test_verification_mutation_cannot_promote_failed_design(self):
        candidates = {}
        def measure(design, operation):
            candidate = assess(design, operation); candidates[design_hash(design)] = candidate
            return candidate
        def render(design):
            if design['elements'][0]['x']:
                candidates[design_hash(design)]['rubric']['gate'] = False
            return verify(design)
        def expand(candidate, budget):
            d = copy.deepcopy(candidate['design']); d['elements'][0]['x'] = 20
            yield 'pixel-invalid', d
        result = run_controller(source(), [('seed', source())], assess=measure, expand=expand,
                                verify=render, mode='single', budget=Budget(seconds=5))[0]
        self.assertEqual(result['design']['elements'][0]['x'], 0)

    def test_cancel_and_no_verified_seed_are_explicit(self):
        cancel = [False]
        def progress(event):
            if event['type'] == 'accepted': cancel[0] = True
        result = run_controller(source(), [('seed', source())], assess=assess, expand=lambda *_: [],
                                verify=verify, progress_callback=progress,
                                budget=Budget(seconds=5, cancellation=lambda: cancel[0]))[0]
        self.assertEqual(result['controller']['stop_reason'], 'cancelled')
        with self.assertRaises(NoValidDesign) as failure:
            run_controller(source(), [('seed', source())], assess=assess, expand=lambda *_: [],
                           verify=lambda _: {'verified': False}, budget=Budget(seconds=5))
        self.assertIsNone(failure.exception.report['incumbent_revision'])

    def test_missing_original_asset_returns_best_layout_without_verification(self):
        calls = []
        def render(design):
            calls.append(design['elements'][0]['x'])
            return {'verified': False, 'error': 'Missing or unsafe image asset: badge.png'}
        def expand(candidate, budget):
            moved = copy.deepcopy(candidate['design']); moved['elements'][0]['x'] = 20
            yield 'move', moved
        result = run_controller(source(), [('seed', source())], assess=assess, expand=expand,
                                verify=render, mode='single', budget=Budget(seconds=5, max_renders=2))[0]
        self.assertEqual(result['design']['elements'][0]['x'], 20)
        self.assertFalse(result['verification']['verified'])
        self.assertFalse(result['verification']['counts_as_success'])
        self.assertEqual(result['controller']['official_verification'], 'missing_original_asset')
        self.assertFalse(result['controller']['counts_as_success'])
        self.assertIsNone(result['controller']['incumbent_revision'])
        self.assertEqual(calls, [0])


if __name__ == '__main__': unittest.main()
