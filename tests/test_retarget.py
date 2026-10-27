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
        designs = propose(source, target, analyze(source), k=3)
        self.assertEqual(len(designs), 3)
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
