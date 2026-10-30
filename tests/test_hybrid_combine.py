import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from hybrid.combine import choose, defects, fix_backgrounds, fix_spacing


def text(id, x, y, w, h, size, z, spacing=0.0):
    return {'id': id, 'type': 'text', 'x': x, 'y': y, 'width': w, 'height': h, 'font_size': size,
            'letter_spacing': spacing, 'content': 'Hello', 'z_index': z}


def image(id, x, y, w, h, z, asset='bg.png'):
    return {'id': id, 'type': 'image', 'x': x, 'y': y, 'width': w, 'height': h, 'asset_id': asset, 'fit': 'fill', 'z_index': z}


def design(w, h, *elements):
    return {'canvas': {'width': w, 'height': h}, 'elements': list(elements)}


class HybridCombineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.assets = Path(self.tmp.name)
        rng = random.Random(0)
        photo = Image.new('RGB', (64, 64))
        photo.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(64 * 64)])
        photo.save(self.assets / 'photo.png')
        Image.new('RGB', (64, 64), (200, 30, 60)).save(self.assets / 'flat.png')

    def tearDown(self):
        self.tmp.cleanup()

    def test_spacing_shrinks_with_the_font_but_never_grows(self):
        source = design(100, 100, text('t', 0, 0, 50, 20, 40, 1, spacing=10))
        smaller = fix_spacing(source, design(50, 50, text('t', 0, 0, 25, 10, 20, 1, spacing=10)))
        larger = fix_spacing(source, design(200, 200, text('t', 0, 0, 100, 40, 80, 1, spacing=10)))
        self.assertEqual(smaller['elements'][0]['letter_spacing'], 5)
        self.assertEqual(larger['elements'][0]['letter_spacing'], 10)

    def test_reshaped_photos_are_cropped_and_flat_fills_still_stretch(self):
        source = design(100, 100, image('p', 0, 0, 100, 100, 0, 'photo.png'), image('f', 0, 0, 100, 100, 1, 'flat.png'))
        out = fix_backgrounds(source, design(400, 100, image('p', 0, 0, 400, 100, 0, 'photo.png'),
                                             image('f', 0, 0, 400, 100, 1, 'flat.png')), self.assets)
        self.assertEqual([e['fit'] for e in out['elements']], ['cover', 'fill'])

    def test_defects_flag_new_collisions_and_text_lifted_off_its_plate(self):
        source = design(100, 100, image('plate', 10, 10, 40, 20, 0), text('a', 12, 12, 30, 10, 8, 1), text('b', 60, 60, 30, 10, 8, 2))
        clean = defects(source, design(100, 100, image('plate', 10, 10, 40, 20, 0), text('a', 12, 12, 30, 10, 8, 1),
                                       text('b', 60, 60, 30, 10, 8, 2)))
        broken = defects(source, design(100, 100, image('plate', 10, 10, 40, 20, 0), text('a', 60, 60, 30, 10, 8, 1),
                                        text('b', 62, 61, 30, 10, 8, 2)))
        self.assertEqual((clean['text_collision'], clean['off_panel']), (0, 0))
        self.assertEqual((broken['text_collision'], broken['off_panel']), (1, 1))

    def test_choose_prefers_fewer_defects_and_survives_a_missing_engine(self):
        source = design(100, 100, text('a', 0, 0, 40, 10, 8, 1), text('b', 0, 50, 40, 10, 8, 2))
        good = design(200, 100, text('a', 0, 0, 40, 10, 8, 1), text('b', 100, 50, 40, 10, 8, 2))
        bad = design(200, 100, text('a', 0, 0, 40, 10, 8, 1), text('b', 2, 1, 40, 10, 8, 2))
        self.assertEqual(choose(source, {'pranay': bad, 'gaurav': good}, self.assets)['chosen'], 'gaurav')
        self.assertEqual(choose(source, {'pranay': None, 'gaurav': bad}, self.assets)['chosen'], 'gaurav')
        with self.assertRaises(ValueError):
            choose(source, {'pranay': None, 'gaurav': None}, self.assets)


if __name__ == '__main__':
    unittest.main()
