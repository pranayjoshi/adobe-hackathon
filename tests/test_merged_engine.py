import json
import unittest
from pathlib import Path

from engine.analysis import analyze
from engine.canonical import Box
from engine.layout import Env
from merged import growth
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
            self.assertTrue(any(p.template.startswith(family) for p in plans(A, *size)), family)

    def test_plans_that_would_make_text_unreadable_are_dropped(self):
        A = analyze(poster())
        made = {p.template: p for p in plans(A, 728, 90)}
        self.assertNotIn('floorplan_row', made)   # the photo with its caption, squeezed into 90 px
        self.assertIn('floorplan_row_free', made)  # the caption beside the photo instead
        caption = made['floorplan_row_free'].patches['caption']['font_size']
        self.assertGreaterEqual(caption, 8)

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

    def test_growth_lets_a_strip_headline_wrap_to_grow_in_a_narrow_column(self):
        from engine.blocks import Ctx, TextBlock
        el = text('h', 0, 0, 1500, 80, 60, 1, 'Kitchen Utensils and Tools for Every Home Cook')  # one long strip line
        block = TextBlock(el, 'headline', 'Headline', 1.0)
        ctx = Ctx(1000, 1500)
        before = TextBlock.layout(block, 2.0, 900, ctx)
        after = growth.text_layout(block, 2.0, 900, ctx)
        size = lambda laid: laid.place(0, 0)[0]['h']['font_size']
        self.assertLessEqual(size(before), 60)   # the source-wrap rule holds it at its source size
        self.assertGreater(size(after), 60)      # allowed to wrap, it grows with the canvas

    def test_growth_keeps_the_source_wrap_when_the_column_is_wide_enough(self):
        from engine.blocks import Ctx, TextBlock
        el = text('h', 0, 0, 300, 80, 60, 1, 'Short')
        block = TextBlock(el, 'headline', 'Headline', 1.0)
        laid = growth.text_layout(block, 1.5, 2000, Ctx(2000, 2000))
        self.assertEqual(laid.place(0, 0)[0]['h']['font_size'], 90)

    def test_growth_cap_follows_the_canvas_area(self):
        A = analyze(poster())
        A.sw, A.sh = 1640.0, 400.0
        env = Env(A, 1000.0, 1500.0)
        self.assertAlmostEqual(growth.s_hi(env), 1.25 * (1000 * 1500 / (1640 * 400)) ** 0.5)
        self.assertEqual(growth.s_hi(Env(A, 728.0, 90.0)), 1.25)  # shrinking targets are unchanged

    def test_floor_plans_are_excused_recomposition_but_not_real_problems(self):
        from merged.resize import rank_key
        source = {'canvas': {'width': 100, 'height': 100}, 'elements': [text('t', 10, 10, 80, 20, 8, 1, 'Hi')]}
        design = {'canvas': {'width': 100, 'height': 100}, 'elements': [text('t', 10, 10, 80, 20, 8, 1, 'Hi')]}
        def candidate(template, **defects):
            found = {'overflow': [], 'collisions': [], 'contrast_regressions': [], 'unreadable': [], 'unsupported_text': [],
                     'scene_displacement': [], **defects}
            return {'template': template, 'design': design, 'valid': True, 'score': {'total': 80},
                    'quality': {'defect_count': sum(len(v) for v in found.values()), 'defects': found},
                    'rubric': {'gate': True, 'tier2': {'defects': [], 'overall': 80}}}
        moved = candidate('floorplan_split', unsupported_text=['t'], scene_displacement=['t'])
        clean = candidate('split_left')
        self.assertGreater(rank_key(source, moved), rank_key(source, clean))   # recomposing is allowed
        collided = candidate('floorplan_split', collisions=[{'ids': ['t']}])
        self.assertLess(rank_key(source, collided), rank_key(source, clean))   # a collision is not


if __name__ == '__main__':
    unittest.main()
