import unittest

import numpy as np

from engine.invariants import passes, validate
from final import pixels
from final.comply import comply
from final.resize import gate, overprinted, vanished


def text(id, x, y, w, h, size, z, content='Hello there', **extra):
    return {'id': id, 'type': 'text', 'x': x, 'y': y, 'width': w, 'height': h, 'font_size': size, 'z_index': z,
            'content': content, 'font_family': 'Solstice Sans', 'color': '#111111', 'text_align': 'left', **extra}


def image(id, x, y, w, h, z):
    return {'id': id, 'type': 'image', 'x': x, 'y': y, 'width': w, 'height': h, 'asset_id': 'solstice_bottle.png', 'fit': 'fill', 'z_index': z}


def source():
    return {'canvas': {'width': 1000, 'height': 1000}, 'elements': [
        image('photo', 100, 100, 400, 200, 1), text('head', 100, 400, 800, 120, 60, 2, letter_spacing=4.0)]}


class ComplyTest(unittest.TestCase):
    def test_a_rule_breaking_layout_comes_back_structurally_clean(self):
        broken = {'canvas': {'width': 728, 'height': 90}, 'elements': [
            image('photo', -40, -10, 400, 120, 1),                     # bleeds off and is stretched
            {**text('head', 380, 5, 340, 80, 40, 2), 'letter_spacing': 1.0, 'color': '#ff0000'}]}   # restyled
        fixed = comply(source(), broken)
        self.assertTrue(passes(validate(source(), fixed)), validate(source(), fixed))
        head = fixed['elements'][1]
        self.assertEqual((head['letter_spacing'], head['color']), (4.0, '#111111'))
        photo = fixed['elements'][0]
        self.assertAlmostEqual(photo['width'] / photo['height'], 2.0)

    def test_a_clean_layout_keeps_its_geometry(self):
        clean = {'canvas': {'width': 1000, 'height': 500}, 'elements': [
            image('photo', 10, 10, 400, 200, 1), text('head', 450, 100, 500, 200, 60, 2, letter_spacing=4.0)]}
        fixed = comply(source(), clean)
        self.assertEqual({k: fixed['elements'][0][k] for k in 'xy'}, {'x': 10, 'y': 10})
        self.assertEqual(fixed['elements'][1]['font_size'], 60)


class GateTest(unittest.TestCase):
    def report(self, design, failed=False, **defects):
        return {'design': design, 'checks': [{'passed': not failed, 'severity': 'error', 'name': 'x'}],
                'quality': {'defects': {'overflow': [], 'contrast_regressions': [], **defects}}}

    def test_new_text_on_text_is_caught_but_the_sources_own_is_not(self):
        s = {'canvas': {'width': 100, 'height': 100}, 'elements': [text('a', 0, 0, 50, 20, 8, 1), text('b', 0, 40, 50, 20, 8, 2)]}
        on_top = {'canvas': s['canvas'], 'elements': [text('a', 0, 0, 50, 20, 8, 1), text('b', 0, 5, 50, 20, 8, 2)]}
        self.assertTrue(overprinted(s, on_top))
        self.assertFalse(overprinted(on_top, on_top))

    def test_the_gate_prefers_legal_layouts_and_never_empties(self):
        s = source()
        good, illegal, overflowing = self.report(s), self.report(s, failed=True), self.report(s, overflow=['head'])
        self.assertEqual(gate(s, [illegal, good, overflowing]), [good])
        self.assertEqual(gate(s, [overflowing]), [overflowing])


class PixelsTest(unittest.TestCase):
    def test_restored_tracking_widens_the_box_instead_of_killing_the_type(self):
        src = {'canvas': {'width': 1000, 'height': 1000},
               'elements': [text('t', 0, 0, 600, 50, 40, 0, content='HOW TO TAKE', letter_spacing=25.0)]}
        narrow = {'canvas': {'width': 970, 'height': 250},
                  'elements': [{'id': 't', 'x': 100, 'y': 100, 'width': 150, 'height': 18, 'font_size': 15}]}
        out = comply(src, narrow)['elements'][0]
        self.assertEqual(out['font_size'], 15)
        self.assertGreater(out['width'], 150)
        self.assertTrue(passes(validate(src, comply(src, narrow))))

    def test_the_gate_prefers_layouts_that_keep_their_text_visible(self):
        src = {'canvas': {'width': 1000, 'height': 1000}, 'elements': [text('t', 0, 0, 600, 50, 40, 0)]}
        def report(size):
            design = {'canvas': {'width': 300, 'height': 250}, 'elements': [text('t', 0, 0, 200, 40, size, 0)]}
            return {'design': design, 'checks': [], 'quality': {'defects': {}}}
        gone, seen = report(1), report(12)
        self.assertEqual(vanished(src, gone['design']), 1)
        self.assertEqual(gate(src, [gone, seen]), [seen])

    def test_a_letterboxed_picture_measures_as_unused(self):
        full = np.zeros((90, 300, 3), dtype=np.uint8) + 255
        full[20:70, 10:290] = (200, 30, 30)
        small = np.zeros((90, 300, 3), dtype=np.uint8) + 255
        small[20:70, 130:170] = (200, 30, 30)
        a, b = pixels.measure(full), pixels.measure(small)
        self.assertLess(a[0], b[0])    # less blank
        self.assertGreater(a[1], b[1])  # wider ink
        self.assertLess(a[2], b[2])    # smaller empty run


if __name__ == '__main__':
    unittest.main()
