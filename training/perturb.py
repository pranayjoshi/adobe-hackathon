"""Negatives built from the guardrails: each perturbation breaks one principle.

  python -m training.perturb --corpus input --out output/training/perturb

Writes `pairs.jsonl` (clean > perturbed, kind=synthetic, for the ranker) and
`detection.json`: how often the rubric flags each violation and how much the
overall score drops. Low detection means that signal is too weak to train on.
"""
import argparse, copy, hashlib, json, random
from collections import defaultdict
from pathlib import Path
from engine.analysis import analyze
from engine.blocks import ButtonBlock, FLOORS
from engine.canonical import element_box
from engine.rubric import safe_zones
from engine.formats import CATALOGUE

SEVERITY = ('small', 'medium', 'large')


def _move(design, ids, dx=0.0, dy=0.0):
    cw, ch = design['canvas']['width'], design['canvas']['height']; by = {e['id']: e for e in design['elements']}
    for i in ids:
        e = by[i]; e['x'] = min(max(0.0, e['x'] + dx), cw - e['width']); e['y'] = min(max(0.0, e['y'] + dy), ch - e['height'])


def perturbations(source, design, A, rng):
    """Yield (principle, severity, expected rubric id prefix, design)."""
    cw, ch = design['canvas']['width'], design['canvas']['height']; by = {e['id']: e for e in design['elements']}
    blocks = [b for b in A.all_blocks() if all(i in by for i in b.ids)]
    texts = [e for e in design['elements'] if e['type'] == 'text' and str(e.get('content', '')).strip()]
    role = lambda i: A.roles.get(i, 'body')
    if blocks:
        b = rng.choice(blocks)
        for sev, dx in zip(SEVERITY, (max(2, .006 * cw), .03 * cw, .10 * cw)):
            d = copy.deepcopy(design); _move(d, b.ids, dx * rng.choice((-1, 1))); yield 'alignment', sev, 'alignment', d
    if texts:
        t = rng.choice(texts)
        for sev, k in zip(SEVERITY, (1.3, 1.7, 2.4)):
            d = copy.deepcopy(design); e = next(x for x in d['elements'] if x['id'] == t['id']); e['font_size'] = max(2, round(e['font_size'] * k))
            yield 'text_fit', sev, 'legibility', d
        body = [e for e in texts if role(e['id']) not in ('headline',)] or texts
        t = rng.choice(body); floor = FLOORS.get(role(t['id']), 9)
        for sev, k in zip(SEVERITY, (.9, .7, .5)):
            d = copy.deepcopy(design); e = next(x for x in d['elements'] if x['id'] == t['id']); e['font_size'] = max(1, int(min(e['font_size'], floor) * k))
            yield 'readability', sev, 'legibility', d
    head = [e for e in texts if role(e['id']) == 'headline']; rest = sorted((e for e in texts if role(e['id']) != 'headline'), key=lambda e: -e['font_size'])
    if head and rest and head[0]['font_size'] > rest[0]['font_size']:
        d = copy.deepcopy(design); x = {e['id']: e for e in d['elements']}
        x[head[0]['id']]['font_size'], x[rest[0]['id']]['font_size'] = rest[0]['font_size'], head[0]['font_size']
        yield 'hierarchy', 'large', 'hierarchy', d
    for b in [b for b in blocks if isinstance(b, ButtonBlock)][:1]:
        label = next(e for e in b.members if e['type'] == 'text')
        for sev, dy in zip(SEVERITY, (20, 40, 60)):
            d = copy.deepcopy(design); _move(d, [label['id']], 0, dy * min(1, ch / 400)); yield 'grouping', sev, 'grouping', d
    if len(texts) >= 2:
        a, b = rng.sample(texts, 2)
        for sev, f in zip(SEVERITY, (.1, .3, .5)):
            d = copy.deepcopy(design); x = {e['id']: e for e in d['elements']}; ea, eb = x[a['id']], x[b['id']]
            ea['x'] = min(max(0, eb['x'] + eb['width'] * (1 - f) * rng.choice((0, .5))), cw - ea['width']); ea['y'] = min(max(0, eb['y'] + eb['height'] * (1 - f)), ch - ea['height'])
            yield 'overlap', sev, 'overlap', d
    for e in design['elements']:
        if e['type'] == 'image' and e.get('fit') == 'fill' and e['id'] not in {x['id'] for x in A.background}:
            for sev, k in zip(SEVERITY, (1.05, 1.1, 1.2)):
                d = copy.deepcopy(design); x = next(y for y in d['elements'] if y['id'] == e['id']); x['width'] = min(cw - x['x'], x['width'] * k)
                yield 'image_integrity', sev, 'S6', d
            break
    # Letterbox: the source uniformly shrunk into the target (looks valid, fits badly).
    d = copy.deepcopy(source); d['canvas'] = dict(design['canvas']); k = min(cw / A.sw, ch / A.sh); ox, oy = (cw - A.sw * k) / 2, (ch - A.sh * k) / 2
    for e in d['elements']:
        e.update(x=e['x'] * k + ox, y=e['y'] * k + oy, width=e['width'] * k, height=e['height'] * k)
        if e['type'] == 'text':
            e['font_size'] = max(1, int(e['font_size'] * k))
    if abs((cw / ch) / (A.sw / A.sh) - 1) > .2:
        yield 'composition', 'large', 'space', d
    d = copy.deepcopy(design); bg = {x['id'] for x in A.background}
    for e in d['elements']:
        if e['id'] not in bg:
            e.update(x=e['x'] * .55, y=e['y'] * .55, width=e['width'] * .55, height=e['height'] * .55)
            if e['type'] == 'text':
                e['font_size'] = max(1, int(e['font_size'] * .55))
    yield 'balance', 'large', 'space', d
    zones = safe_zones(cw, ch)
    if zones and texts:
        t = rng.choice(texts); z = max(zones, key=lambda z: z.area); d = copy.deepcopy(design)
        e = next(x for x in d['elements'] if x['id'] == t['id']); e['x'] = min(max(0, z.x), cw - e['width']); e['y'] = min(max(0, z.y), ch - e['height'])
        yield 'safe_zone', 'medium', 'space', d


def _flags(report, prefix):
    if prefix.startswith('S'):
        return any(f['id'] == prefix for f in report['tier1']['failures'])
    t2 = report['tier2'] or {}
    return any(x['id'].split('.')[0] == prefix for x in t2.get('defects', []) + t2.get('warnings', []))


def run(corpus, out, formats=None, seed=7, limit=None):
    from engine.pipeline import generate
    from training.reward import assess
    from training.export import split_of
    from training.families import refuse_qa_corpus
    refuse_qa_corpus(corpus)
    out.mkdir(parents=True, exist_ok=True); rng = random.Random(seed); pairs = []; stats = defaultdict(lambda: {'n': 0, 'flagged': 0, 'drop': 0.0, 'ranked_lower': 0})
    targets = formats or [(w, h) for _, w, h in CATALOGUE]
    for path in sorted(corpus.glob('*.json'))[:limit]:
        task = json.loads(path.read_text()); source = task.get('source', task); family = task.get('family') or (task.get('metadata') or {}).get('source_id') or path.stem
        A = analyze(source); held_out = split_of(family) == 'test'  # audited, but never emitted as training pairs
        for w, h in targets:
            target = {'width': w, 'height': h}
            best = generate(source, target, top_k=1, options={'render': False})[0]
            if not best['rubric']['passed']:
                continue  # negatives need a clean positive
            clean = {k: best[k] for k in ('design', 'score', 'diagnostics', 'rubric')}
            for principle, sev, expect, design in perturbations(source, best['design'], A, rng):
                bad = assess(source, design, A, target); key = f'{principle}/{sev}'; s = stats[key]; s['n'] += 1
                s['flagged'] += _flags(bad['rubric'], expect)
                overall = (bad['rubric']['tier2'] or {}).get('overall', 0); s['drop'] += clean['rubric']['tier2']['overall'] - overall
                s['ranked_lower'] += (not bad['rubric']['gate']) or overall < clean['rubric']['tier2']['overall']
                if held_out:
                    continue
                pair = hashlib.sha256(json.dumps([path.stem, w, h, key]).encode()).hexdigest()[:16]
                pairs.append({'pair': pair, 'family': family, 'kind': 'synthetic', 'principle': principle, 'severity': sev, 'target': target,
                              'decision': 'a', 'a': clean, 'b': {k: bad[k] for k in ('design', 'score', 'diagnostics', 'rubric')}})
            print(path.stem, w, h, len(pairs), flush=True)
    report = {k: {'n': v['n'], 'detection_rate': round(v['flagged'] / v['n'], 3), 'ranked_lower_rate': round(v['ranked_lower'] / v['n'], 3),
                  'mean_overall_drop': round(v['drop'] / v['n'], 2)} for k, v in sorted(stats.items())}
    (out / 'pairs.jsonl').write_text('\n'.join(json.dumps(p) for p in pairs) + '\n')
    (out / 'detection.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--corpus', type=Path, default=Path('input')); ap.add_argument('--out', type=Path, default=Path('output/training/perturb'))
    ap.add_argument('--formats', help='comma list like 1080x1080,728x90'); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--limit', type=int); a = ap.parse_args()
    formats = [tuple(map(int, f.split('x'))) for f in a.formats.split(',')] if a.formats else None
    print(json.dumps(run(a.corpus, a.out, formats, a.seed, a.limit), indent=2))
