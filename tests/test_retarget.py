import json
import math
import unittest
from pathlib import Path

from engine.retarget.features import (TASK_ARRANGE, TEXT, base_box, centered, decode_target, encode_source,
                                      encode_target, N_FEATURES, N_GLOBAL)
from engine.retarget.infer import weights_path


class RetargetFeatureTests(unittest.TestCase):
    def test_target_round_trip_is_within_one_bin(self):
        src = {'type': TEXT, 'x': 100, 'y': 200, 'w': 300, 'h': 50, 'font_size': 30, 'text': 'Sale', 'lines': 1}
        tgt = dict(src, x=40, y=20, w=420, h=36, font_size=22)
        out = decode_target(encode_target(tgt, src, 1000, 1000, 728, 90), src, 1000, 1000, 728, 90)
        self.assertLess(abs(out['x'] + out['w'] / 2 - (tgt['x'] + tgt['w'] / 2)), 728 * 0.01)
        self.assertLess(abs(math.log(out['w'] / tgt['w'])), 0.03)
        self.assertLess(abs(math.log(out['font_size'] / tgt['font_size'])), 0.03)

    def test_identity_target_is_the_stretch_baseline(self):
        src = {'type': 2, 'x': 10, 'y': 10, 'w': 200, 'h': 100}
        self.assertEqual(encode_target(src, src, 500, 500, 500, 500)[:4], [128, 128, 128, 128])

    def test_source_encoding_shapes_and_arrange_task(self):
        els = [{'type': 3, 'x': 0, 'y': 0, 'w': 800, 'h': 600},
               {'type': TEXT, 'x': 50, 'y': 60, 'w': 300, 'h': 80, 'font_size': 40, 'text': 'Hello', 'lines': 1}]
        types, feats, glob = encode_source(els, 800, 600, 300, 600)
        self.assertEqual(feats.shape, (2, N_FEATURES)); self.assertEqual(glob.shape, (N_GLOBAL,))
        moved = centered(els, 800, 600, 300, 600)
        self.assertEqual((moved[0]['w'], moved[0]['h']), (300, 600))  # background covers the target
        self.assertAlmostEqual(moved[1]['x'] + moved[1]['w'] / 2, 150)
        self.assertEqual(encode_source(moved, 300, 600, 300, 600, TASK_ARRANGE)[2][3:].tolist(), [0, 0, 1])

    def test_base_box_keeps_graphics_inside_extreme_targets(self):
        b = base_box({'type': 2, 'x': 0, 'y': 100, 'w': 1000, 'h': 800}, 1000, 1000, 728, 90)
        self.assertLessEqual(b['h'], 90 + 1e-6); self.assertAlmostEqual(b['w'] / b['h'], 1000 / 800)


@unittest.skipUnless(weights_path().exists(), 'retarget weights not exported')
class RetargetProposalTests(unittest.TestCase):
    def test_proposals_preserve_elements_and_hard_rules(self):
        from engine.analysis import analyze
        from engine.retarget.propose import propose
        case = json.loads(Path('input/chameleon_sample_level2_core.json').read_text())
        source, target = case['source'], {'width': 728, 'height': 90}
        tagged = propose(source, target, analyze(source), k=3)
        self.assertEqual([t for t, _ in tagged if t != '0_bound'], ['0', '1', '2'])
        designs = [d for _, d in tagged]
        by_id = {e['id']: e for e in source['elements']}
        for d in designs:
            self.assertEqual(d['canvas'], target)
            self.assertEqual([e['id'] for e in d['elements']], [e['id'] for e in source['elements']])
            for e in d['elements']:
                s = by_id[e['id']]
                self.assertEqual({k: v for k, v in e.items() if k not in ('x', 'y', 'width', 'height', 'font_size', 'fit')},
                                 {k: v for k, v in s.items() if k not in ('x', 'y', 'width', 'height', 'font_size', 'fit')})
                if e['type'] == 'image' and s['width'] < 1000:
                    self.assertAlmostEqual(e['width'] / e['height'], s['width'] / s['height'], delta=0.03 * s['width'] / s['height'])
                if e['type'] == 'text':
                    self.assertGreaterEqual(e['x'], -0.01); self.assertLessEqual(e['x'] + e['width'], 728.01)


if __name__ == '__main__':
    unittest.main()


class DeoverlapTests(unittest.TestCase):
    def _source(self):
        text = lambda i, y, s: {'id': i, 'type': 'text', 'x': 100, 'y': y, 'width': 800, 'height': s * 1.3, 'z_index': 2,
                                'content': i.upper() + ' COPY', 'font_family': 'Inter', 'font_size': s, 'color': '#000000',
                                'text_align': 'center'}
        return {'canvas': {'width': 1000, 'height': 1000}, 'elements': [
            {'id': 'bg', 'type': 'shape', 'x': 0, 'y': 0, 'width': 1000, 'height': 1000, 'z_index': 0,
             'shape_type': 'rectangle', 'fill_color': '#ffffff'},
            {'id': 'card', 'type': 'shape', 'x': 300, 'y': 600, 'width': 400, 'height': 300, 'z_index': 1,
             'shape_type': 'rectangle', 'fill_color': '#ff0000'},
            text('title', 100, 80), text('sub', 260, 40), text('note', 700, 30)]}

    def test_removes_new_collisions_and_keeps_source_order(self):
        from engine.analysis import analyze
        from engine.quality import collisions
        from engine.retarget.deoverlap import deoverlap
        source = self._source(); A = analyze(source)
        design = json.loads(json.dumps(source)); design['canvas'] = {'width': 1200, 'height': 600}
        by = {e['id']: e for e in design['elements']}
        by['bg'].update(width=1200, height=600)
        by['title'].update(x=200, y=120, width=800); by['sub'].update(x=200, y=150, width=800)  # piled up
        by['card'].update(x=400, y=380, width=400, height=200); by['note'].update(x=450, y=440, width=300)
        self.assertTrue(collisions(source, design, A))
        out = deoverlap(source, design, A)
        self.assertEqual(collisions(source, out, A), [])
        o = {e['id']: e for e in out['elements']}
        self.assertLessEqual(o['title']['y'] + o['title']['height'], o['sub']['y'])  # title stays above
        for e in out['elements']:
            self.assertGreaterEqual(e['x'], -0.01); self.assertLessEqual(e['x'] + e['width'], 1200.01)
            self.assertGreaterEqual(e['y'], -0.01); self.assertLessEqual(e['y'] + e['height'], 600.01)

    def test_leaves_collision_free_layouts_untouched(self):
        from engine.analysis import analyze
        from engine.retarget.deoverlap import deoverlap
        source = self._source()
        self.assertEqual(deoverlap(source, source, analyze(source)), source)

    def test_stack_may_reflow_into_a_row_on_a_wide_canvas(self):
        from engine.retarget.deoverlap import _relation
        photo, panel = [0, 0, 1000, 600], [0, 620, 1000, 380]          # source: photo above panel
        now_photo, now_panel = [0, 0, 833, 500], [700, 120, 1000, 380]  # 2000x500: side by side, touching
        axis, photo_first, allowed = _relation(photo, panel, now_photo, now_panel, 0.0)
        # Stacking cannot fit in 500px; keep the order on the horizontal axis (above -> left).
        self.assertEqual((axis, photo_first, allowed), (0, True, None))
        # When stacking is still the cheap fix, the source axis is kept.
        self.assertEqual(_relation(photo, panel, [0, 0, 1000, 600], [0, 580, 1000, 380], 0.0)[:2], (1, True))


class BackingTests(unittest.TestCase):
    def _source(self):
        return {'canvas': {'width': 600, 'height': 600}, 'elements': [
            {'id': 'bg', 'type': 'shape', 'x': 0, 'y': 0, 'width': 600, 'height': 600, 'z_index': 0,
             'shape_type': 'rectangle', 'fill_color': '#ffffff'},
            {'id': 'badge', 'type': 'shape', 'x': 380, 'y': 380, 'width': 180, 'height': 120, 'z_index': 1,
             'shape_type': 'rectangle', 'fill_color': '#cc2222'},
            {'id': 'dot', 'type': 'shape', 'x': 390, 'y': 390, 'width': 4, 'height': 4, 'z_index': 2,
             'shape_type': 'ellipse', 'fill_color': '#ffffff'},
            {'id': 'title', 'type': 'text', 'x': 40, 'y': 60, 'width': 500, 'height': 80, 'z_index': 3,
             'content': 'BIG TITLE', 'font_family': 'Inter', 'font_size': 60, 'color': '#000000'},
            {'id': 'label', 'type': 'text', 'x': 400, 'y': 420, 'width': 140, 'height': 40, 'z_index': 4,
             'content': 'SALE', 'font_family': 'Inter', 'font_size': 30, 'color': '#ffffff'}]}

    def test_copy_attaches_to_its_smallest_backing_and_specks_stay_free(self):
        from engine.retarget.deoverlap import backed
        self.assertEqual(backed(self._source()), [['badge', 'label']])

    def test_copy_stays_on_its_badge_when_predictions_pull_them_apart(self):
        from engine.analysis import analyze
        from engine.retarget.propose import materialize, records
        source = self._source(); A = analyze(source); recs = records(source, A)
        target = {'width': 300, 'height': 900}
        pred = {'bg': (0, 0, 300, 900, 0), 'badge': (20, 700, 200, 133, 0), 'dot': (30, 710, 4, 4, 0),
                'title': (20, 60, 260, 60, 40), 'label': (150, 100, 120, 40, 30)}  # label far from badge
        p = [dict(zip(('x', 'y', 'w', 'h', 'font_size'), pred[r['id']])) for r in recs]
        out = {e['id']: e for e in materialize(source, target, A, recs, p)['elements']}
        b, t = out['badge'], out['label']
        self.assertTrue(b['x'] <= t['x'] and t['x'] + t['width'] <= b['x'] + b['width'] + 0.5)
        self.assertTrue(b['y'] <= t['y'] and t['y'] + t['height'] <= b['y'] + b['height'] + 0.5)


class LearnedGuardTests(unittest.TestCase):
    def test_copy_margin_is_four_percent_unless_the_source_bled(self):
        from engine.retarget.deoverlap import margin
        self.assertAlmostEqual(margin([100, 100, 200, 50], 1000, 1000, 1600, 400), 16.0)
        self.assertAlmostEqual(margin([5, 100, 200, 50], 1000, 1000, 1600, 400), 2.0)  # bleed kept, scaled

    def _pair(self):
        el = lambda i, x, y, w, h, z: {'id': i, 'type': 'shape', 'x': x, 'y': y, 'width': w, 'height': h, 'z_index': z,
                                       'shape_type': 'rectangle', 'fill_color': '#000000'}
        source = {'canvas': {'width': 1000, 'height': 1000},
                  'elements': [el('photo', 0, 0, 1000, 600, 0), el('panel', 0, 620, 1000, 380, 1)]}
        return source, json.loads(json.dumps(source))

    def test_panel_slid_over_a_photo_is_new_occlusion(self):
        from engine.pipeline import _new_occlusion
        source, design = self._pair()
        self.assertEqual(_new_occlusion(source, design), [])
        design['elements'][1].update(y=200)
        self.assertEqual(_new_occlusion(source, design), ['photo'])

    def test_collapsed_content_share(self):
        from engine.pipeline import _content_share
        source, design = self._pair()
        self.assertAlmostEqual(_content_share(source), 1.0)
        for e in design['elements']:
            e.update(x=e['x'] * 0.3 + 300, y=e['y'] * 0.3 + 300, width=e['width'] * 0.3, height=e['height'] * 0.3)
        self.assertLess(_content_share(design), 0.5 * _content_share(source))


class SolverTests(unittest.TestCase):
    def _sample(self):
        case = json.loads(Path('input/chameleon_sample_level2_core.json').read_text())
        return case['source']

    def test_text_options_cover_single_words_and_tracking(self):
        from engine.retarget.solver import text_options
        word = {'id': 't', 'type': 'text', 'content': 'Merry', 'font_family': 'Inter', 'font_size': 40, 'width': 200, 'height': 50}
        opts = text_options(word, 40, 4)
        self.assertEqual([o.lines for o in opts], [1])
        long = dict(word, content='Spring cleaning day at the park this Sunday')
        self.assertGreater(len(text_options(long, 30, 6)), 2)
        tracked = dict(long, letter_spacing=8)
        o1, o2 = text_options(long, 30, 1)[0], text_options(tracked, 30, 1)[0]
        self.assertGreater(o2.b - o1.b, 8 * 10)  # tracking is a constant width term, not scaled

    @unittest.skipUnless(weights_path().exists(), 'retarget weights not exported')
    def test_solved_layout_passes_the_hard_rules(self):
        from engine.analysis import analyze
        from engine.invariants import passes, validate
        from engine.retarget.infer import predict_dist
        from engine.retarget.propose import propose, records
        from engine.retarget.solver import Problem, solve
        source = self._sample(); A = analyze(source); recs = records(source, A)
        for target in ({'width': 1080, 'height': 1080}, {'width': 1200, 'height': 628}):
            W, H = source['canvas']['width'], source['canvas']['height']
            boxes, probs, _ = predict_dist(recs, W, H, target['width'], target['height'], k=1)
            warm = dict(propose(source, target, A, k=1))['0']
            sol = solve(Problem(source, target, A, recs), boxes[0], probs[0], time_limit=0.5, hint=warm, workers=2)
            self.assertIsNotNone(sol.design)
            self.assertTrue(passes(validate(source, sol.design)), validate(source, sol.design))
            self.assertEqual([e['id'] for e in sol.design['elements']], [e['id'] for e in source['elements']])

    def test_strip_composer_lays_content_along_the_strip(self):
        from engine.analysis import analyze
        from engine.canonical import rotated_extent
        from engine.retarget.propose import records
        from engine.retarget.solver import Problem
        from engine.retarget.strip import compose, is_strip, long_axis
        self.assertTrue(is_strip({'width': 800, 'height': 90})); self.assertTrue(is_strip({'width': 90, 'height': 800}))
        self.assertFalse(is_strip({'width': 1200, 'height': 628}))
        self.assertEqual((long_axis({'width': 800, 'height': 90}), long_axis({'width': 90, 'height': 800})), (0, 1))
        source = self._sample(); A = analyze(source)
        target = {'width': 970, 'height': 90}
        d = compose(Problem(source, target, A, records(source, A)))
        self.assertEqual(len(d['elements']), len(source['elements']))
        for e in d['elements']:
            b = rotated_extent(e)
            self.assertGreaterEqual(b.x, -0.5); self.assertLessEqual(b.x1, 970.5)
            self.assertGreaterEqual(b.y, -0.5); self.assertLessEqual(b.y1, 90.5)

    def test_strips_keep_a_lower_content_share_floor(self):
        from engine.pipeline import _collapse_floor
        self.assertEqual(_collapse_floor({'canvas': {'width': 800, 'height': 90}}), .3)
        self.assertEqual(_collapse_floor({'canvas': {'width': 1080, 'height': 1080}}), .5)


class PostprocDefaultTests(unittest.TestCase):
    def test_solver_post_processing_is_on_unless_switched_off(self):
        import os
        from unittest import mock
        from engine.retarget.propose import postproc
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CHAMELEON_POSTPROC', None)
            self.assertEqual(postproc({}), 'solver')
            self.assertEqual(postproc({'postproc': 'off'}), 'off')
            os.environ['CHAMELEON_POSTPROC'] = 'off'
            self.assertEqual(postproc({}), 'off')
            self.assertEqual(postproc({'postproc': 'solver'}), 'solver')

    @unittest.skipUnless(weights_path().exists(), 'retarget weights not exported')
    def test_solver_stops_starting_rounds_after_its_deadline(self):
        import time
        from engine.analysis import analyze
        from engine.retarget.infer import predict_dist
        from engine.retarget.propose import propose, records
        from engine.retarget.solver import Problem, solve
        source = json.loads(Path('input/chameleon_sample_level2_core.json').read_text())['source']
        A = analyze(source); recs = records(source, A); target = {'width': 1200, 'height': 628}
        boxes, probs, _ = predict_dist(recs, 1080, 1080, 1200, 628, k=1)
        warm = dict(propose(source, target, A, k=1))['0']
        t = time.monotonic()
        sol = solve(Problem(source, target, A, recs), boxes[0], probs[0], time_limit=0.5, hint=warm, workers=2, deadline=time.monotonic())
        self.assertLess(time.monotonic() - t, 4.0)
        self.assertIsNotNone(sol)


class StripAdapterTests(unittest.TestCase):
    def test_gate_is_off_below_three_to_one_and_full_from_four_to_one(self):
        from engine.retarget.infer import strip_gate
        self.assertEqual(strip_gate(1200, 628), 0.0)
        self.assertEqual(strip_gate(1500, 500), 0.0)
        self.assertEqual(strip_gate(970, 90), 1.0); self.assertEqual(strip_gate(90, 800), 1.0)
        self.assertEqual(strip_gate(1600, 400), 1.0)
        self.assertTrue(0 < strip_gate(350, 100) < 1)

    @unittest.skipUnless(weights_path().exists(), 'retarget weights not exported')
    def test_gate_zero_leaves_the_base_model_unchanged(self):
        import tempfile
        import numpy as np
        from engine.retarget.features import encode_source
        from engine.retarget.infer import Model
        from engine.retarget.propose import records
        from engine.analysis import analyze
        base = Model(weights_path())
        rng = np.random.default_rng(0)
        d = base.cfg['d']
        arrays = {}
        for i in range(base.cfg['dec_layers']):
            for name in ('attn.q', 'attn.o', 'xattn.q', 'xattn.o'):
                arrays[f'dec.{i}.{name}.lora_A'] = rng.normal(0, .5, (4, d)).astype(np.float32)
                arrays[f'dec.{i}.{name}.lora_B'] = rng.normal(0, .5, (d, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adapter.npz'
            np.savez(path, **arrays)
            adapted = Model(weights_path(), adapter=path)
        source = json.loads(Path('input/chameleon_sample_level2_core.json').read_text())['source']
        recs = records(source, analyze(source))
        t, f, g = encode_source(recs, 1080, 1080, 970, 90)
        same = base.sample(t, f, g, k=1)
        self.assertTrue((adapted.sample(t, f, g, k=1, gate=0.0) == same).all())
        self.assertFalse((adapted.sample(t, f, g, k=1, gate=1.0) == same).all())


class StripContrastTests(unittest.TestCase):
    def test_white_copy_off_its_dark_panel_moves_back_onto_it(self):
        from engine.analysis import analyze
        from engine.retarget.contrast import keep_contrast
        from engine.quality import text_contrast
        panel = {'id': 'p', 'type': 'shape', 'x': 0, 'y': 0, 'width': 400, 'height': 300, 'fill_color': 'rgba(20,20,20,1)', 'z_index': 1}
        text = {'id': 't', 'type': 'text', 'x': 40, 'y': 120, 'width': 300, 'height': 50, 'content': 'Sale', 'font_family': 'Inter',
                'font_size': 40, 'color': 'rgba(255,255,255,1)', 'text_align': 'left', 'z_index': 2}
        bg = {'id': 'bg', 'type': 'shape', 'x': 0, 'y': 0, 'width': 1200, 'height': 300, 'fill_color': 'rgba(255,255,255,1)', 'z_index': 0}
        source = {'canvas': {'width': 1200, 'height': 300}, 'elements': [bg, panel, text]}
        strip = {'canvas': {'width': 800, 'height': 90}, 'elements': [
            dict(bg, width=800, height=90), dict(panel, x=0, y=0, width=200, height=90), dict(text, x=500, y=25, width=120, height=40, font_size=30)]}
        A = analyze(source)
        self.assertLess(text_contrast(strip)[0]['t'], 3)
        fixed = keep_contrast(source, strip, A)
        t = next(e for e in fixed['elements'] if e['id'] == 't')
        self.assertGreaterEqual(text_contrast(fixed)[0]['t'], 3)
        self.assertLessEqual(t['x'] + t['width'], 200.5)
        self.assertEqual((t['font_size'], t['color'], t['width']), (30, text['color'], 120))


class RenderPoolTests(unittest.TestCase):
    def test_warm_worker_renders_match_fresh_processes_and_survive_a_crash(self):
        import os
        from unittest import mock
        from PIL import ImageChops
        from engine import quality, render_pool
        design = json.loads(Path('input/chameleon_sample_level2_core.json').read_text())['source']
        with mock.patch.dict(os.environ, {'CHAMELEON_RENDER_POOL': '0'}):
            quality._CACHE.clear(); fresh = quality.verify_design(design, max_size=320)
        quality._CACHE.clear(); warm = quality.verify_design(design, max_size=320)
        if not fresh['verified']:
            self.skipTest(fresh.get('error', 'renderer unavailable'))
        self.assertTrue(warm['verified'], warm.get('error'))
        self.assertIsNone(ImageChops.difference(fresh['image'], warm['image']).getbbox())
        for w in list(render_pool._idle):
            w.proc.kill(); w.proc.wait()  # a crashed worker is replaced on the next render
        quality._CACHE.clear(); again = quality.verify_design(design, max_size=320)
        self.assertTrue(again['verified'], again.get('error'))


class AppealTests(unittest.TestCase):
    @unittest.skipUnless(Path('engine/models/appeal_features.joblib').exists(), 'appeal model not trained')
    def test_appeal_score_is_a_probability_and_prefers_the_intact_design(self):
        from engine.analysis import analyze
        from engine.retarget.appeal import score, source_info
        case = json.loads(Path('input/chameleon_sample_level2_core.json').read_text())
        source = case['source']; A = analyze(source); si = source_info(source, A)
        good = score(source, source, A, si)
        bad = json.loads(json.dumps(source))
        for e in bad['elements']:
            if e['type'] == 'text':
                e['x'] = 0; e['y'] = 0  # all copy piled into the corner
        worse = score(source, bad, A, si)
        self.assertTrue(0.0 <= good <= 1.0 and 0.0 <= worse <= 1.0)
        self.assertGreater(good, worse)
