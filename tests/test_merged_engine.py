import json
import unittest
from pathlib import Path

from engine.analysis import analyze
from engine.canonical import Box
from engine.layout import Env
from merged.floorplan import fit_weighted, plans, units
from merged.resize import scale_spacing, wasted_canvas

ROOT = Path(__file__).resolve().parents[1]


def text(id, x, y, w, h, size, z, content):
    return {'id': id, 'type': 'text', 'x': x, 'y': y, 'width': w, 'height': h, 'font_size': size, 'z_index': z,
            'content': content, 'font_family': 'Solstice Sans', 'color': '#111111', 'text_align': 'left'}


def image(id, x, y, w, h, z, asset='solstice_bottle.png'):
    return {'id': id, 'type': 'image', 'x': x, 'y': y, 'width': w, 'height': h, 'asset_id': asset, 'fit': 'fill', 'z_index': z}


def poster():
    """A portrait poster: photo with a caption printed on it, headline, a rule under it, legal line."""
    return {'canvas': {'width': 1000, 'height': 1500}, 'elements': [
        image('photo', 100, 100, 800, 700, 1, 'solstice_bg_gradient.png'),
        text('caption', 150, 650, 500, 60, 40, 2, 'On the photo'),
        text('headline', 100, 900, 800, 120, 100, 3, 'Big Headline'),
        image('rule', 100, 1030, 400, 6, 4),
        text('legal', 100, 1350, 800, 40, 20, 5, 'Terms and conditions apply to every offer'),
    ]}


class MergedEngineTest(unittest.TestCase):
    def test_role_weighted_fit_shrinks_minor_copy_before_the_headline(self):
        A = analyze(poster())
        env = Env(A, 728.0, 90.0)
        blocks = [b for b in A.blocks if b.role in ('headline', 'legal', 'body', 'subhead')]
        fit = fit_weighted(env, blocks, Box(0, 0, 700, 400))
        sizes = {}
        for b, laid in zip(blocks, fit.laid):
            patches, _ = laid.place(0, 0)
            for eid, p in patches.items():
                if 'font_size' in p:
                    sizes[eid] = p['font_size'] / next(e['font_size'] for e in poster()['elements'] if e['id'] == eid)
        self.assertGreater(sizes['headline'], sizes['legal'])

    def test_units_keep_text_on_its_photo_and_rules_with_their_text(self):
        A = analyze(poster())
        us, loose = units(A)
        holder = {eid: u.id for u in us for eid in u.ids}
        self.assertEqual(holder['caption'], holder['photo'])
        if 'rule' in holder:  # the analysis may already have attached it
            self.assertIn(holder['rule'], (holder['headline'], holder['photo']))

    def test_every_shape_family_gets_a_plan(self):
        A = analyze(poster())
        for size, family in (((1080, 1920), 'floorplan_column'), ((1600, 900), 'floorplan_split'), ((728, 90), 'floorplan_row')):
            self.assertIn(family, [p.template for p in plans(A, *size)])

    def test_wasted_canvas_flags_a_letterboxed_background(self):
        source = {'canvas': {'width': 100, 'height': 100}, 'elements': [image('bg', 0, 0, 100, 100, 0), text('t', 10, 10, 50, 10, 8, 1, 'Hi')]}
        filled = {'canvas': {'width': 400, 'height': 100}, 'elements': [image('bg', 0, 0, 400, 100, 0), text('t', 10, 10, 200, 10, 8, 1, 'Hi')]}
        boxed = {'canvas': {'width': 400, 'height': 100}, 'elements': [image('bg', 150, 0, 100, 100, 0), text('t', 160, 10, 50, 10, 8, 1, 'Hi')]}
        self.assertEqual(wasted_canvas(source, filled), 0)
        self.assertGreater(wasted_canvas(source, boxed), 0)

    def test_spacing_only_shrinks_with_the_font(self):
        source = {'canvas': {'width': 10, 'height': 10}, 'elements': [{**text('t', 0, 0, 5, 5, 40, 0, 'Hi'), 'letter_spacing': 8}]}
        small = {'canvas': {'width': 5, 'height': 5}, 'elements': [{**text('t', 0, 0, 5, 5, 20, 0, 'Hi'), 'letter_spacing': 8}]}
        large = {'canvas': {'width': 20, 'height': 20}, 'elements': [{**text('t', 0, 0, 5, 5, 80, 0, 'Hi'), 'letter_spacing': 8}]}
        self.assertEqual(scale_spacing(source, small)['elements'][0]['letter_spacing'], 4)
        self.assertEqual(scale_spacing(source, large)['elements'][0]['letter_spacing'], 8)


if __name__ == '__main__':
    unittest.main()
