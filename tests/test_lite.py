import unittest

from fastapi.testclient import TestClient

from lite.app import app


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


if __name__ == '__main__':
    unittest.main()
