"""Mine designer-made resizes: the same design published at several sizes.

  python -m training.families CORPUS_DIR --out output/training/families.jsonl

Elements are matched by type and content (text), asset (image) or shape
style, never by ID. A pair is kept only when every element matches one to
one and the reference passes rubric Tier 0-1 as a resize of the source, so
it is a valid stage-one example (same copy, fonts, assets and styles).
"""
import argparse, copy, json, hashlib
from collections import defaultdict
from pathlib import Path
from engine.canonical import MUTABLE_FIELDS
from engine.rubric import tier0, tier1


SPLITS = ('train', 'validation', 'calibration', 'test')


def refuse_qa_corpus(corpus):
    if (Path(corpus) / 'QA_ONLY').exists():
        raise ValueError('QA_ONLY workspace contains agent test designs, never training data')


def resolve_family(record, fallback=None):
    """Trajectory/resizes inherit original provenance; never split by revision ID."""
    family = record.get('source_family') or record.get('family') or (record.get('metadata') or {}).get('source_id') or fallback
    if family is None:
        raise ValueError('source family provenance required')
    return str(family)


def split_of(family, manifest=None):
    family = str(family)
    if manifest is not None:
        aliases = manifest.get('aliases', {}); seen = set()
        while family in aliases:
            if family in seen: raise ValueError('cyclic family alias')
            seen.add(family); family = str(aliases[family])
        if family in manifest.get('excluded_families', []):
            raise ValueError('family excluded from training: ' + family)
        split = manifest.get('families', {}).get(family)
        if split not in SPLITS: raise ValueError('family missing from frozen split manifest: ' + family)
        return split
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'


def immutable_signature(source):
    values = [{k:v for k,v in e.items() if k not in MUTABLE_FIELDS and k != 'id'} for e in source.get('elements', [])]
    return hashlib.sha256(json.dumps(sorted(values, key=lambda x:json.dumps(x,sort_keys=True)), sort_keys=True).encode()).hexdigest()


def build_split_manifest(records):
    """Union exact immutable-content and provenance families before partitioning.

    Perceptual near-duplicates still require reviewed alias additions. This is
    deliberately conservative: it does not guess that similar-looking copy is identical.
    """
    parents = {}; fingerprints = {}; aliases = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x: parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parents[max(a,b)] = min(a,b)
    for fallback, record in records:
        family = resolve_family(record, fallback); find(family)
        source = record.get('source', record)
        # All immutable fields participate; mutable geometry never does.
        values = [{k:v for k,v in e.items() if k not in MUTABLE_FIELDS and k != 'id'} for e in source.get('elements', [])]
        signature = immutable_signature(source)
        if values:
            if signature in fingerprints: union(family, fingerprints[signature])
            else: fingerprints[signature] = family
        aliases[str(fallback)] = family
    aliases = {k:find(v) for k,v in aliases.items() if k != find(v)}
    aliases.update({k:find(k) for k in parents if k != find(k)})
    families = {}
    for k in parents:
        root = find(k); bucket = int(hashlib.sha256(root.encode()).hexdigest()[:8],16) % 100
        families[root] = 'train' if bucket < 70 else 'validation' if bucket < 80 else 'calibration' if bucket < 85 else 'test'
    return {'schema_version':1,'families':families,'aliases':aliases,'excluded_families':[],
            'policy':'provenance + exact immutable content union; sha256 cluster 70/10/5/15 train/validation/calibration/test; reviewed near-duplicate aliases required'}


def _key(e):
    if e['type'] == 'text':
        return ('text', e.get('content'), e.get('font_family'), e.get('color'))
    if e['type'] == 'image':
        return ('image', e.get('asset_id'))
    return ('shape', e.get('shape_type'), e.get('fill_color'), e.get('stroke_color'))


def signature(design):
    return tuple(sorted(map(str, (_key(e) for e in design['elements']))))


def match(a, b):
    """Reference design using a's elements with b's geometry, or None if ambiguous."""
    keys = defaultdict(list)
    for e in b['elements']:
        keys[_key(e)].append(e)
    if any(len(v) > 1 for v in keys.values()) or len(a['elements']) != len(b['elements']):
        return None
    out = copy.deepcopy(a); out['canvas'] = {'width': int(b['canvas']['width']), 'height': int(b['canvas']['height'])}
    for e in out['elements']:
        other = keys.get(_key(e))
        if not other:
            return None
        e.update({k: other[0][k] for k in MUTABLE_FIELDS if k in other[0]})
    return out


def mine(corpus):
    refuse_qa_corpus(corpus)
    groups = defaultdict(list)
    for path in sorted(Path(corpus).glob('*.json')):
        try:
            record = json.loads(path.read_text()); design = record.get('source', record)
            if design.get('elements'):
                groups[signature(design)].append((path.stem, record, design))
        except (ValueError, KeyError, TypeError):
            continue
    pairs = []
    for members in groups.values():
        for na, ra, a in members:
            for nb, rb, b in members:
                if na == nb or a['canvas'] == b['canvas']:
                    continue
                ref = match(a, b)
                if ref and tier0(a, ref)['passed'] and tier1(a, ref, ref['canvas'])['passed']:
                    family = 'content-' + hashlib.sha256(repr(signature(a)).encode()).hexdigest()[:24]
                    pairs.append({'family': str(family), 'source_file': na, 'reference_file': nb, 'source': a,
                                  'target_canvas': ref['canvas'], 'reference': ref, 'origin': 'designer'})
    return pairs


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('corpus', type=Path); ap.add_argument('--out', type=Path, default=Path('output/training/families.jsonl')); a = ap.parse_args()
    pairs = mine(a.corpus); a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(''.join(json.dumps(p) + '\n' for p in pairs))
    print(json.dumps({'pairs': len(pairs), 'families': len({p['family'] for p in pairs}), 'out': str(a.out)}))
