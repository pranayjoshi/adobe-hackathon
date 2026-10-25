"""Release gate from the Eval Rubric. Exit code 0 only if every check passes.

  python -m evaluation.release_gate output/evaluation/RUN [--baseline output/evaluation/OLD]

RUN should be a `--render --offline` benchmark. If `judge compare` has written
RUN/comparison.json, Tier 3 is checked. Pairwise evidence is required for promotion;
missing evidence fails explicitly even when no baseline argument is supplied.
"""
import argparse, json, sys
from pathlib import Path
from engine.canonical import ROOT


def check(run, baseline=None):
    data = json.loads((run / 'results.json').read_text()); s = data['summary']; cases = data['cases']; out = []
    add = lambda name, ok, detail='': out.append({'check': name, 'passed': bool(ok), 'detail': detail})
    keys = [c.get('case') for c in cases]
    add('Complete denominator', bool(cases) and len(keys) == len(set(keys)) and s.get('cases') == len(cases), 'empty, duplicate or missing case rows')
    errors = [c.get('case') for c in cases if c.get('error') or not c.get('rubric')]
    add('Every case assessed', not errors, ', '.join(str(x) for x in errors))
    missing = [c.get('case') for c in cases if not c.get('rendered') or not c.get('preview') or not (run / c['preview']).is_file() or not (run / (c['case'] + '.json')).is_file()]
    add('Required image and editable artifacts', not missing, ', '.join(str(x) for x in missing))
    add('Tier 0: zero disqualifiers', s['tier0_failures'] == 0, ', '.join(s.get('tier0_cases', [])[:10]))
    add('Tier 1: zero structural failures', not s['tier1_failures'], json.dumps(s['tier1_failures']))
    add('Offline: no network attempts', s.get('offline') and not s.get('network_attempts'), 'run with --offline' if not s.get('offline') else str(s.get('network_attempts')))
    add('Rendered: exact JSON renders', s['pixels'] and s.get('render_failures', 0) == 0, 'run with --render' if not s['pixels'] else f"{s.get('render_failures', 0)} failures")
    bad = lambda prefix: [c['case'] for c in cases if any(x.startswith(prefix) for x in c.get('rubric', {}).get('defects', []))]
    add('Tier 2: no new or worsened overflow', not bad('legibility.overflow'), ', '.join(bad('legibility.overflow')[:10]))
    add('Tier 2: zero text-on-text overlap', not bad('overlap.text'), ', '.join(bad('overlap.text')[:10]))
    add('Tier 2: zero broken rigid groups', not bad('grouping.rigid'), ', '.join(bad('grouping.rigid')[:10]))
    head = [c['case'] for c in cases if c.get('bucket') != 'extreme' and 'legibility.floor' in c.get('rubric', {}).get('defects', [])]
    add('Tier 2: text above floors on non-extreme tasks', not head, ', '.join(head[:10]))
    if baseline:
        prior_data = json.loads((baseline / 'results.json').read_text()); old = prior_data['summary']
        old_keys = {c.get('case') for c in prior_data['cases']}
        add('Paired identical cases', set(keys) == old_keys, 'baseline and challenger case identities differ')
        add('Paired frozen corpus', s.get('corpus_hash') is not None and s.get('corpus_hash') == old.get('corpus_hash'), 'source bytes differ or missing corpus identity')
        add('Paired identical evaluator', bool(s.get('evaluation_identity')) and s.get('evaluation_identity') == old.get('evaluation_identity'), 'raw rubric means cannot establish output improvement with different or unknown evaluators')
        for b, block in s['tier2_by_bucket'].items():
            prior = old.get('tier2_by_bucket', {}).get(b)
            if prior and prior['overall_mean'] is not None and block['overall_mean'] is not None:
                add(f'Tier 2 no regression ({b})', block['overall_mean'] >= prior['overall_mean'] - 1.0, f"{block['overall_mean']} vs {prior['overall_mean']}")
    comparison = run / 'comparison.json'
    if comparison.exists():
        c = json.loads(comparison.read_text())['summary']
        add('Tier 3: pairwise no worse in every bucket', c['no_worse_everywhere'], json.dumps(c['tally']))
    else:
        add('Tier 3: pairwise comparison present', False, 'run python -m evaluation.judge compare RUN BASELINE')
    adapters = [p for p in (ROOT / 'engine/models').rglob('*') if p.suffix in ('.gguf', '.safetensors', '.bin')]
    pub = ROOT / 'engine/models/publication.json'
    urls = json.loads(pub.read_text()) if pub.exists() else {}
    unpublished = [p.name for p in adapters if not str(urls.get(p.name, '')).startswith('https://')]
    add('Model weights publicly accessible', not unpublished, 'list public URLs in engine/models/publication.json: ' + ', '.join(unpublished) if unpublished else '')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); ap.add_argument('--baseline', type=Path); a = ap.parse_args()
    results = check(a.run, a.baseline)
    for r in results:
        print('PASS' if r['passed'] else 'FAIL', r['check'], ('- ' + r['detail']) if r['detail'] and not r['passed'] else '')
    sys.exit(0 if all(r['passed'] for r in results) else 1)
