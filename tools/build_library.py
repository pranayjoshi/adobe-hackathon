"""Export training-only role/geometry slots; images and text never enter the library.

Legacy (v1, engine/models/layouts.json):
    python -m tools.build_library <corpus_dir> [--output engine/models/layouts.json]

Versioned v2 (engine/models/layouts.v2.json, plan 05):
    python -m tools.build_library --v2 --labels work/role-data/labels.jsonl \
        [--suite docs/individual-update-plans/visual-suite.v1.json] [--output engine/models/layouts.v2.json]

v2 records layout_id, source family, split, provenance, ratio, role counts,
normalized slots and grouping. The visual-suite sources (their exact-copy
families, dataset source ids, and any geometry/copy near-duplicate) and the
validation split are excluded at construction; feature normalization uses
the training partition only. No network access.
"""
import argparse, hashlib, json, re
from pathlib import Path


def build_v1(corpus: Path, output: Path):
    from engine.analysis import analyze
    from engine.retrieval import export_layout
    library = []; seen = set()
    for path in corpus.glob('*.json'):
        t = json.loads(path.read_text())
        if t.get('split') != 'train': continue
        d = t.get('source', t); entry = export_layout(d, analyze(d)); key = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        if key not in seen: library.append(entry); seen.add(key)
    if not library: raise SystemExit('No verified train-split designs; library not overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(library)); print(len(library), 'training layouts exported')


def _tokens(source):
    return {w for e in source['elements'] for w in re.findall(r'[a-z0-9]{3,}', str(e.get('content', '')).lower())}


def build_v2(labels: Path, suite: Path, output: Path, repo: Path, split_manifest: Path | None = None):
    from engine.providers import library as L
    from training.families import build_split_manifest
    manifest = json.loads(suite.read_text())
    suite_sources = []
    for s in manifest['sources']:
        p = repo / s['path']
        if p.exists():
            suite_sources.append(json.loads(p.read_text())[s.get('source_field', 'source')])
    held_f = {f for f in (L.text_family(s) for s in suite_sources) if f}
    held_s = set(L.VISUAL_SUITE_SOURCE_IDS) | {s['source_id'] for s in manifest['sources'] if s.get('source_id')}
    suite_sigs = set(); suite_tokens = []
    for s in suite_sources:
        e = L.build_entry(s, family='suite')
        if e: suite_sigs.add(L._signature(e))
        suite_tokens.append(_tokens(s))
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    frozen = (json.loads(split_manifest.read_text()) if split_manifest else
              build_split_manifest([(str(n), r) for n, r in enumerate(rows)]))
    entries, near_suite = [], 0
    revs = {}
    man = labels.with_suffix('.manifest.json')
    if man.exists(): revs = json.loads(man.read_text()).get('datasets', {})
    for row in rows:
        src = row['source']
        toks = _tokens(src)
        if any(t and len(toks & t) / max(1, len(toks | t)) >= 0.5 for t in suite_tokens):
            near_suite += 1; held_f.add(row['family']); continue
        e = L.build_entry(src, row.get('roles'), family=row['family'], source_id=row.get('source_id', ''),
                          provenance={'dataset': 'cyberagent/crello', 'revision': revs.get('cyberagent/crello'),
                                      'labels': 'DesignAsCode machine-distilled themes (weak)', 'label_kind': row.get('kind', 'weak')})
        if e is None: continue
        if L._signature(e) in suite_sigs:
            near_suite += 1; held_f.add(row['family']); continue
        entries.append(e)
    lib = L.assemble(entries, held_out_families=held_f, held_out_source_ids=held_s,
                     build={'builder': 'tools/build_library.py --v2', 'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest(),
                            'suite_sha256': hashlib.sha256(suite.read_bytes()).hexdigest(), 'datasets': revs,
                            'rows': len(rows), 'suite_near_duplicates': near_suite,
                            'split_rule': frozen['policy'],
                            'split_manifest_sha256': hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()},
                     split_manifest=frozen)
    L.validate_library(lib)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lib, separators=(',', ':'), sort_keys=True))
    output.with_suffix('.splits.json').write_text(json.dumps(frozen,indent=2))
    print(json.dumps(lib['build'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', type=Path, nargs='?')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--v2', action='store_true')
    ap.add_argument('--labels', type=Path, default=Path('work/role-data/labels.jsonl'))
    ap.add_argument('--suite', type=Path, default=Path('docs/individual-update-plans/visual-suite.v1.json'))
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--split-manifest', type=Path)
    a = ap.parse_args()
    if a.v2:
        build_v2(a.labels, a.suite, a.output or Path('engine/models/layouts.v2.json'), a.repo, a.split_manifest)
    else:
        if a.corpus is None: ap.error('corpus is required for the legacy v1 library')
        build_v1(a.corpus, a.output or Path('engine/models/layouts.json'))
