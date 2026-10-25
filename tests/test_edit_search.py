"""Coordinated edit search (guide 07): trap, budget/cancellation, cycles, materializer contract."""
import copy
import unittest

from engine.analysis import analyze
from engine.contracts import DesignRevision
from engine.edit_operators import SearchBudget, expand
from engine.invariants import passes, validate
from engine.materialize import apply_edit, proposal_from_design
from engine.quality import collisions
from engine.refine import refine


def text(eid, x, w, z, content='Sale', **kw):
    return {'id': eid, 'type': 'text', 'content': content, 'font_family': 'Solstice Sans', 'font_size': 24,
            'font_weight': 'bold', 'color': '#111111', 'x': x, 'y': 0, 'width': w, 'height': 100, 'z_index': z,
            'text_align': 'left', 'line_height': 1.2, **kw}


def trap():
    """A locked badge at the left edge; two full-height labels overlap and have nowhere to go alone."""
    source = {'canvas': {'width': 200, 'height': 100}, 'elements': [
        {'id': 'pill', 'type': 'shape', 'shape_type': 'rectangle', 'fill_color': '#ffcc00', 'x': 0, 'y': 0,
         'width': 60, 'height': 100, 'z_index': 0, 'group_id': 'badge'},
        text('pill_t', 5, 50, 1, 'New', group_id='badge'),
        text('a', 64, 64, 2), text('b', 134, 64, 3)]}
    seed = copy.deepcopy(source)
    by = {e['id']: e for e in seed['elements']}
    by['a'].update(x=60, width=80); by['b'].update(x=120, width=80)
    locks = {i: {k: by[i][k] for k in ('x', 'y', 'width', 'height')} for i in ('pill', 'pill_t')}
    return source, seed, locks


def fake_assessor(source, A, locks, calls=None, score=None):
    """Pipeline-shaped assessment: validity, locks and new block collisions only."""
    def assess(design):
        if calls is not None:
            calls.append(design)
        by = {e['id']: e for e in design['elements']}
        ok = passes(validate(source, design)) and all(abs(by[i][k] - v) < .6 for i, r in locks.items() for k, v in r.items())
        col = collisions(source, design, A)
        ids = [i for c in col for i in c['ids']]
        overall = score(design) if score else 100.0 - 20 * len(col)
        return {'design': design, 'valid': ok, 'template': 'fixture',
                'rubric': {'gate': True, 'tier2': {'defects': [{'id': 'overlap.collision', 'ids': ids, 'severity': 'defect'}] if col else [],
                                                   'warnings': [], 'overall': overall, 'scores': {}, 'stats': {}}},
                'quality': {'defect_count': len(col), 'defects': {'unreadable': [], 'overflow': [], 'collisions': col, 'contrast_regressions': []}},
                'diagnostics': {'collisions': col}}
    return assess


class EditSearchTests(unittest.TestCase):
    def test_coordinated_neighbourhood_solves_two_edit_trap_and_keeps_lock(self):
        source, seed, locks = trap(); A = analyze(source)
        assess = fake_assessor(source, A, locks)
        start = assess(seed)
        self.assertTrue(start['diagnostics']['collisions'])
        climbed, _ = refine(start, assess, source, A, locks, mode='hill_climb')
        self.assertTrue(climbed['diagnostics']['collisions'], 'single-step hill climb should be stuck')
        searched, steps = refine(start, assess, source, A, locks, mode='beam_lns', budget=SearchBudget(max_evals=60))
        self.assertFalse(searched['diagnostics']['collisions'])
        self.assertTrue(searched['valid'])
        self.assertTrue(any(s.get('family', '').startswith('compound:') for s in steps))
        by = {e['id']: e for e in searched['design']['elements']}
        for i, rect in locks.items():
            for k, v in rect.items():
                self.assertAlmostEqual(by[i][k], v, delta=.5)
        self.assertEqual([e.get('content') for e in searched['design']['elements']], [e.get('content') for e in source['elements']])

    def test_cancellation_mid_expansion_returns_best_assessed_incumbent(self):
        source, seed, locks = trap(); A = analyze(source)
        marks = {61: 90.0, 62: 50.0, 63: 99.0}   # second trial is worse; third never runs
        assess = fake_assessor(source, A, locks, score=lambda d: marks.get(round(d['elements'][2]['x']), 10.0))
        calls = []
        def shifts(ctx, r):
            for x in (61, 62, 63):
                d = copy.deepcopy(r['design']); d['elements'][2]['x'] = x
                calls.append(x); yield f'shift {x}', d
        n = []
        budget = SearchBudget(cancelled=lambda: len(n) >= 2)
        counted = lambda d: (n.append(1), assess(d))[1]
        best, steps = refine(assess(seed), counted, source, A, locks, mode='beam_lns', budget=budget, operators=[shifts])
        self.assertEqual(round(best['design']['elements'][2]['x']), 61)
        self.assertEqual(len(n), 2)
        self.assertEqual(steps[-1]['refine_search']['stop_reason'], 'cancelled')

    def test_cycling_operator_is_bounded_by_geometry_dedupe(self):
        source, seed, locks = trap(); A = analyze(source)
        calls = []
        assess = fake_assessor(source, A, locks, calls)
        def flip(ctx, r):
            for dx in (10, -10):
                d = copy.deepcopy(r['design'])
                d['elements'][2]['x'] += dx
                yield 'flip', d
        start = assess(seed); calls.clear()
        refine(start, assess, source, A, locks, mode='beam_lns', operators=[flip], max_rounds=50)
        # States are x in {50, 60, 70, ...}: each distinct geometry is assessed at most once,
        # and the depth cap (6) bounds the walk even without any budget.
        xs = [round(d['elements'][2]['x']) for d in calls]
        self.assertEqual(len(xs), len(set(xs)))
        self.assertLessEqual(len(xs), 2 * 6)

    def test_expand_outputs_survive_the_materializer(self):
        from unittest.mock import patch
        source, seed, locks = trap(); A = analyze(source)
        candidate = fake_assessor(source, A, locks)(seed)
        out = list(expand(candidate, SearchBudget(max_evals=100), source=source, analysis=A, locks=locks))
        self.assertTrue(out)
        with patch('engine.constraints.solve', side_effect=AssertionError('solver disabled')) as solver:
            self.assertTrue(list(expand(candidate, source=source, analysis=A, locks=locks, allow_solver=False)))
            solver.assert_not_called()
        rev = DesignRevision.create(candidate['design'])
        for name, design in out:
            apply_edit(source, rev, proposal_from_design(rev, design, name), locks)

    def test_shared_contract_budget_counts_assessments_and_rejects_tiny_lock_drift(self):
        from engine.contracts import Budget
        source, seed, locks = trap(); A = analyze(source)
        assess = fake_assessor(source, A, locks)
        calls = []
        def shifts(ctx, r):
            drift = copy.deepcopy(r['design']); drift['elements'][0]['x'] += .001
            yield 'illegal lock drift', drift
            for x in (61, 62, 63):
                d = copy.deepcopy(r['design']); d['elements'][2]['x'] = x
                yield 'move text', d
        budget = Budget(seconds=10, max_edits=1)
        counted = lambda d: (calls.append(d), assess(d))[1]
        _, trace = refine(assess(seed), counted, source, A, locks, mode='beam_lns', budget=budget, operators=[shifts])
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.edits, 1)
        self.assertEqual(trace[-1]['refine_search']['stop_reason'], 'edit_budget')
        self.assertEqual(calls[0]['elements'][0]['x'], seed['elements'][0]['x'])

    def test_cancelled_expansion_and_zero_limit_do_no_operator_work(self):
        source, seed, locks = trap(); A = analyze(source)
        candidate = fake_assessor(source, A, locks)(seed)
        for kw in ({'budget': SearchBudget(cancelled=lambda: True)}, {'max_edits': 0}):
            self.assertEqual(list(expand(candidate, source=source, analysis=A, locks=locks, **kw)), [])

    def test_repair_does_not_move_locked_group_during_containment(self):
        from engine.repair import repair
        source, seed, locks = trap(); A = analyze(source)
        # A previously accepted lock may be infeasible after a target resize.
        # Repair must expose that failure instead of silently moving the lock.
        locks = copy.deepcopy(locks);locks['pill']['x'] = -10;locks['pill_t']['x'] = -5
        out, _ = repair(source, seed, A, locks)
        by = {e['id']:e for e in out['elements']}
        for eid,rect in locks.items():
            self.assertEqual({k:by[eid][k] for k in rect},rect)


if __name__ == '__main__':
    unittest.main()
