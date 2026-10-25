"""Lean behavior checks for the provider contract, diversity and fit-aware retrieval (plans 03/05)."""
import copy, json, math, os, tempfile, unittest
from pathlib import Path
from unittest import mock

from engine.analysis import analyze
from engine.invariants import passes, validate
from engine.providers import (Initializer, Proposal, ProviderResult, SourceFrameProvider, TemplateProvider,
                              build_context, fingerprint, legacy_proposals, mmr_select, run_providers)
from engine.providers import library as L
from engine.providers.retrieval_fit import assign, build_units, materialize, retrieve_fit_aware

ROOT = Path(__file__).resolve().parents[1]


def text(i, x, y, w=300, h=60, fs=40, content='Some headline text', z=2):
    return dict(id=i, type='text', x=x, y=y, width=w, height=h, font_family='Solstice Sans', font_size=fs,
                content=content, color='rgba(255,255,255,1)', z_index=z)


def image(i, asset, x, y, w, h, z=1):
    return dict(id=i, type='image', asset_id=asset, x=x, y=y, width=w, height=h, z_index=z)


def small_source():
    return {'canvas': {'width': 800, 'height': 800}, 'elements': [
        image('bg', 'solstice_bg_gradient.png', 0, 0, 800, 800, z=0),
        image('hero', 'solstice_bottle.png', 250, 60, 300, 300),
        text('head', 80, 420, 640, 90, 56, 'Golden hour sparkling sale'),
        text('body', 120, 540, 560, 60, 22, 'Small-batch botanical infusions, bottled at sunset.')]}


class _Raises:
    name, cost = 'raises', 1
    def propose(self, context, budget=None):
        context.source['elements'][0]['width'] = 0
        context.analysis.roles.clear()
        raise RuntimeError('model crashed after mutating its local context')


class _Wrong:
    name, cost = 'wrong_type', 1
    def propose(self, context, budget=None): return [context.source]


class _Emits:
    name, cost = 'emits', 1
    def __init__(self, mutate): self.mutate = mutate
    def propose(self, context, budget=None):
        d = {'canvas': dict(context.target), 'elements': copy.deepcopy(context.source['elements'])}
        self.mutate(d)
        return ProviderResult(self.name, 'ok', (Proposal(self.name, 'bad', 'Bad', '', d),))


def _set(i, k, v):
    def f(d): d['elements'][i][k] = v
    return f


class ProviderContractTests(unittest.TestCase):
    def test_malformed_or_failed_provider_cannot_suppress_deterministic_output(self):
        src = small_source(); target = {'width': 1200, 'height': 400}
        ctx = build_context(src, target)
        cases = [(_Raises(), 'failed', None), (_Wrong(), 'failed', None),
                 (_Emits(lambda d: d['elements'].append({**d['elements'][0], 'id': 'ghost'})), 'empty', 'unknown_id'),
                 (_Emits(lambda d: d['elements'].append(copy.deepcopy(d['elements'][1]))), 'empty', 'duplicate_id'),
                 (_Emits(lambda d: d['elements'].pop()), 'empty', 'missing_id'),
                 (_Emits(lambda d: d['elements'].reverse()), 'empty', 'element_order'),
                 (_Emits(_set(1, 'x', math.nan)), 'empty', 'non_finite'),
                 (_Emits(_set(1, 'width', True)), 'empty', 'boolean_number'),
                 (_Emits(_set(2, 'y', -40.0)), 'empty', 'negative_origin'),
                 (_Emits(_set(2, 'color', 'rgba(0,0,0,1)')), 'empty', 'non_allowlisted_field'),
                 (_Emits(lambda d: d['canvas'].update(width=999)), 'empty', 'canvas_mismatch')]
        for bad, status, code in cases:
            with self.subTest(provider=bad.name, code=code):
                bad_r, frame_r = run_providers(ctx, [bad, SourceFrameProvider()], schedule='given')
                self.assertEqual(bad_r.status, status)
                self.assertFalse(bad_r.proposals)
                if code: self.assertEqual([r['code'] for r in bad_r.rejections], [code])
                else: self.assertTrue(bad_r.error)
                self.assertEqual(frame_r.status, 'ok'); self.assertEqual(len(frame_r.proposals), 1)
                seeds = list(Initializer([bad, SourceFrameProvider()])(ctx))
                self.assertEqual([op for op, _ in seeds], ['initialize/source_frame/source_frame'])
        self.assertEqual(ctx.source, src)  # immutable source snapshot

    def test_legacy_adapters_reproduce_pipeline_proposals(self):
        from engine import pipeline
        src = json.loads((ROOT / 'input/chameleon_sample_level1_warmup.json').read_text())['source']
        target = {'width': 1200, 'height': 628}
        seen = []
        def record(source, design, A, locks=()):
            seen.append(copy.deepcopy(design)); return design, []
        with mock.patch('engine.repair.repair', record):
            pipeline.generate(src, target, options={'solver': False, 'refine': False, 'render': False}, all_candidates=True)
        proposals, _, results = legacy_proposals(build_context(src, target))
        expected = [d for t, _, _, d in proposals if t not in ('identity', 'source_frame')]
        self.assertEqual(seen, expected)
        self.assertEqual({r.provider: r.status for r in results}['vlm'], 'skipped')


class DiversityTests(unittest.TestCase):
    def test_diversity_follows_geometry_and_gates_dominate(self):
        src = small_source(); A = analyze(src)
        a = copy.deepcopy(src)
        moved = copy.deepcopy(src)
        for e in moved['elements']:
            if e['id'] == 'hero': e.update(x=20, y=420, width=200, height=200)
            if e['id'] == 'head': e.update(x=260, y=420, width=520)
        items = [{'name': 'stack', 'design': a, 'valid': 1, 'q': 1.0},
                 {'name': 'split_left', 'design': copy.deepcopy(a), 'valid': 1, 'q': 0.99},  # same geometry, new name
                 {'name': 'stack', 'design': moved, 'valid': 1, 'q': 0.9}]                 # same name, new geometry
        fp = lambda it: fingerprint(it['design'], A)
        pick = mmr_select(items, 2, quality=lambda it: it['q'], fingerprint_of=fp, gate=lambda it: it['valid'])
        self.assertEqual([p is items[0] or p is items[2] for p in pick], [True, True])
        items[2]['valid'] = 0  # distinct but invalid: never chosen over a valid near-duplicate
        pick = mmr_select(items, 2, quality=lambda it: it['q'], fingerprint_of=fp, gate=lambda it: it['valid'])
        self.assertNotIn(items[2], pick)

    def test_same_size_identity_not_displaced_by_more_providers(self):
        from engine.pipeline import generate
        src = small_source()
        env = {'CHAMELEON_PROPOSAL_PORTFOLIO': 'diverse', 'CHAMELEON_RETRIEVAL_VERSION': 'fit_aware'}
        with mock.patch.dict(os.environ, env):
            out = generate(src, dict(src['canvas']), options={'render': False, 'solver': False, 'refine': False})
        self.assertEqual(out[0]['template'], 'identity')


def _lib(entries, **kw):
    return L.assemble(entries, held_out_families=kw.get('held', ()), held_out_source_ids=(), build={'test': True})


def _entry(slots, ratio, family):
    src = {'canvas': {'width': ratio * 1000, 'height': 1000}, 'elements': []}
    for n, s in enumerate(slots):
        el = {'id': f's{n}', 'type': s.get('type', 'text'), 'x': s['x'] * ratio * 1000, 'y': s['y'] * 1000,
              'width': s['w'] * ratio * 1000, 'height': s['h'] * 1000, 'z_index': n + 1,
              'font_size': s.get('fs', 40), 'content': s.get('content', 'x' * 20)}
        src['elements'].append(el)
    roles = {f's{n}': s['role'] for n, s in enumerate(slots)}
    e = L.build_entry(src, roles, family=family, source_id=family)
    e['split'] = 'train'
    return e


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.src = small_source(); self.A = analyze(self.src)
        self.target = {'width': 1200, 'height': 400}

    def test_unequal_slots_keep_every_block_and_reject_impossible_semantics(self):
        units = build_units(self.src, self.A)
        many = _entry([{'role': 'image', 'type': 'image', 'x': .05, 'y': .1, 'w': .3, 'h': .8},
                       {'role': 'headline', 'x': .4, 'y': .1, 'w': .5, 'h': .3},
                       {'role': 'body', 'x': .4, 'y': .5, 'w': .5, 'h': .2},
                       {'role': 'legal', 'x': .4, 'y': .8, 'w': .5, 'h': .1},
                       {'role': 'image', 'type': 'image', 'x': .9, 'y': .1, 'w': .08, 'h': .2}], 3.0, 'fam_many')
        result, _ = assign(units, many, self.target, self.A)
        self.assertEqual(sorted(result), list(range(len(units))))  # complete coverage, spare slots stay empty
        for i, j in result.items():
            self.assertEqual(units[i].cls == 'visual', many['slots'][j]['cls'] == 'visual')
        design = materialize(self.src, self.target, self.A, units, many, result)
        self.assertTrue(passes(validate(self.src, design)))
        text_only = _entry([{'role': 'headline', 'x': .1, 'y': .1, 'w': .8, 'h': .3},
                            {'role': 'body', 'x': .1, 'y': .5, 'w': .8, 'h': .3}], 3.0, 'fam_text')
        with self.assertRaises(ValueError) as err:  # a hero image can never fill a copy slot
            assign(units, text_only, self.target, self.A)
        self.assertIn(str(err.exception), ('insufficient_slots', 'impossible_semantic_assignment'))

    def test_measured_text_fit_prefers_readable_headline(self):
        # Same ratio. "close" mirrors the source boxes but squeezes the headline
        # into a sliver; "roomy" moves things yet leaves the headline space.
        close = _entry([{'role': 'image', 'type': 'image', 'x': .31, 'y': .02, 'w': .38, 'h': .45},
                        {'role': 'headline', 'x': .46, 'y': .52, 'w': .08, 'h': .05, 'fs': 56},
                        {'role': 'body', 'x': .15, 'y': .58, 'w': .7, 'h': .4, 'fs': 22}], 3.0, 'fam_close')
        roomy = _entry([{'role': 'image', 'type': 'image', 'x': .03, 'y': .05, 'w': .3, 'h': .9},
                        {'role': 'headline', 'x': .38, 'y': .1, 'w': .58, 'h': .45, 'fs': 56},
                        {'role': 'body', 'x': .38, 'y': .62, 'w': .58, 'h': .3, 'fs': 22}], 3.0, 'fam_roomy')
        target = {'width': 600, 'height': 200}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lib.json'; path.write_text(json.dumps(_lib([close, roomy])))
            designs, report = retrieve_fit_aware(self.src, target, self.A, library_path=path)
        self.assertTrue(designs)
        self.assertEqual([s['layout_id'] for s in report['selected']], [roomy['layout_id']])
        self.assertIn({'layout_id': close['layout_id'], 'reason': 'tiny_headline'},
                      [{k: r[k] for k in ('layout_id', 'reason')} for r in report['rejected']])
        self.assertGreaterEqual(report['selected'][0]['headline_ratio'], .98)
        head = next(e for e in designs[0]['elements'] if e['id'] == 'head')
        self.assertGreaterEqual(head['font_size'], 14)
        # ...even though the rejected arrangement sits closer to the source geometry.
        gap = lambda e: sum(abs((s['x'] + s['w'] / 2) - (0.5)) + abs((s['y'] + s['h'] / 2) - 465 / 800) for s in e['slots'] if s['role'] == 'headline')
        self.assertLess(gap(close), gap(roomy))

    def test_contaminated_or_incompatible_library_is_rejected_and_templates_survive(self):
        good = _lib([_entry([{'role': 'headline', 'x': .1, 'y': .1, 'w': .8, 'h': .3}], 2.0, 'fam_a')])
        L.validate_library(good)
        def held(lib): lib['held_out']['families'] = ['fam_a']
        def split(lib): lib['entries'][0]['split'] = 'validation'; lib['entries_sha256'] = L.entries_hash(lib['entries'])
        def emb(lib): lib['embedding'] = {'encoder': 'google/siglip2-base-patch16-224', 'preprocessing': 'v0'}
        def schema(lib): lib['schema'] = 'chameleon-layout-library/1'
        def tamper(lib): lib['entries'][0]['ratio'] = 9.0
        def content(lib): lib['entries'][0]['slots'][0]['content'] = 'copy'; lib['entries_sha256'] = L.entries_hash(lib['entries'])
        def frozen_split(lib): lib['family_split_manifest'] = {'families': {'fam_a': 'test'}, 'aliases': {}}
        def nonfinite(lib): lib['normalization']['std'][0] = float('nan')
        for mutate, reason in [(held, 'held_out_family'), (split, 'non_training_split'), (emb, 'incompatible_embedding'),
                               (schema, 'incompatible_schema'), (tamper, 'integrity_mismatch'), (content, 'content_in_library'),
                               (frozen_split,'family_split_mismatch'),(nonfinite,'incompatible_normalization')]:
            with self.subTest(reason=reason):
                lib = copy.deepcopy(good); mutate(lib)
                with self.assertRaises(L.LibraryError) as err: L.validate_library(lib)
                self.assertEqual(err.exception.reason, reason)
        from engine.pipeline import generate
        with tempfile.TemporaryDirectory() as tmp:
            bad = copy.deepcopy(good); held(bad); path = Path(tmp) / 'bad.json'; path.write_text(json.dumps(bad))
            env = {'CHAMELEON_RETRIEVAL_VERSION': 'fit_aware', 'CHAMELEON_LAYOUT_LIBRARY_V2': str(path)}
            with mock.patch.dict(os.environ, env):
                out = generate(self.src, self.target, options={'render': False, 'solver': False, 'refine': False}, all_candidates=True)
        self.assertTrue(any(c['valid'] for c in out))
        self.assertFalse(any(c['template'].startswith('retrieved') for c in out))
        self.assertIn({'retrieval': 'invalid library ignored'}, out[0]['trace'][-1]['events'])

    def test_shipped_library_excludes_visual_suite(self):
        lib = L.load_library(ROOT / 'engine/models/layouts.v2.json')
        suite = json.loads((ROOT / 'docs/individual-update-plans/visual-suite.v1.json').read_text())
        fams = {L.text_family(json.loads((ROOT / s['path']).read_text())['source']) for s in suite['sources']}
        ids = {s.get('source_id') for s in suite['sources']}
        self.assertFalse({e['source_family'] for e in lib['entries']} & fams)
        self.assertFalse({e['source_id'] for e in lib['entries']} & ids)
        self.assertEqual({e['split'] for e in lib['entries']}, {'train'})


if __name__ == '__main__':
    unittest.main()
