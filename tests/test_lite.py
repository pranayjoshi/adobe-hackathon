import unittest

from fastapi.testclient import TestClient

from lite.app import app, same_shape, scale_design


class LiteAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_page_and_presets(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        presets = self.client.get('/api/presets').json()
        self.assertTrue(presets and all({'width', 'height', 'label'} <= set(p) for p in presets))

    def test_rejects_malformed_designs(self):
        bad = [{'x': 1}, {'canvas': {'width': 100, 'height': 100}, 'elements': []},
               {'canvas': {'width': 100, 'height': 100}, 'elements': [{'id': 'a', 'type': 'text', 'x': 0, 'y': 0, 'width': 10, 'height': 10}]},
               {'canvas': {'width': 100, 'height': 100}, 'elements': [{'id': 'a', 'type': 'image', 'asset_id': '../../etc/passwd', 'x': 0, 'y': 0, 'width': 10, 'height': 10}]}]
        for design in bad:
            self.assertEqual(self.client.post('/api/resize', json={'source': design, 'sizes': [{'width': 300, 'height': 250}]}).status_code, 422)

    def test_unknown_job(self):
        self.assertEqual(self.client.get('/api/jobs/000000000000').status_code, 404)
        self.assertEqual(self.client.get('/api/jobs/../x').status_code, 404)


class SameShapeTests(unittest.TestCase):
    source = {'canvas': {'width': 1000, 'height': 500}, 'elements': [
        {'id': 't', 'type': 'text', 'x': 100, 'y': 50, 'width': 400, 'height': 100, 'content': 'Hi',
         'font_family': 'Inter', 'font_size': 40, 'color': '#000', 'letter_spacing': 2},
        {'id': 'r', 'type': 'shape', 'shape_type': 'rectangle', 'fill_color': '#fff', 'x': 0, 'y': 0,
         'width': 1000, 'height': 500, 'corner_radius': 10}]}

    def test_shape_tolerance(self):
        self.assertTrue(same_shape(self.source, {'width': 2000, 'height': 1000}))
        self.assertTrue(same_shape(self.source, {'width': 1600, 'height': 790}))  # 1.3% off
        self.assertFalse(same_shape(self.source, {'width': 1600, 'height': 900}))

    def test_same_size_is_unchanged(self):
        self.assertEqual(scale_design(self.source, {'width': 1000, 'height': 500}), self.source)

    def test_scaled_keeps_layout(self):
        out = scale_design(self.source, {'width': 2000, 'height': 1000})
        t, r = out['elements']
        self.assertEqual(out['canvas'], {'width': 2000, 'height': 1000})
        self.assertEqual((t['x'], t['y'], t['width'], t['height'], t['font_size'], t['letter_spacing']), (200, 100, 800, 200, 80, 4))
        self.assertEqual((r['width'], r['height'], r['corner_radius']), (2000, 1000, 20))
        self.assertEqual(self.source['elements'][0]['font_size'], 40)  # source untouched


if __name__ == '__main__':
    unittest.main()
