"""Run: python -m evaluation.benchmark --corpus input --name samples --render [--offline].
Held-out Crello: --corpus output/crello/designs --split test (family = metadata.source_id,
split as training/export.py split_of, so these families are never trained on).

Reports the Eval Rubric tiers separately (docs/chameleon-design-guardrails-and-eval-rubric.md):
Tier 0/1 failure counts, Tier 2 mean / p10 / defect rate per dimension sliced
by aspect bucket, and runtime. Tier 3 is `python -m evaluation.judge`.
"""
import argparse, hashlib, json, platform, socket, statistics, time
from collections import defaultdict
from pathlib import Path
from engine.pipeline import generate
from engine.canonical import ASSETS_DIR
from engine.formats import CATALOGUE, aspect_bucket, direction
from engine.rubric import WEIGHTS, summary as rubric_summary

FORMATS = [(w, h) for _, w, h in CATALOGUE]


class _NoNetwork:
    """Rubric D6: any connection attempt during the run is a failure."""
    def __enter__(self):
        self.attempts = []; self._connect = socket.socket.connect
        def refuse(sock, address):
            self.attempts.append(str(address)); raise OSError('network disabled for offline benchmark')
        socket.socket.connect = refuse; return self

    def __exit__(self, *exc):
        socket.socket.connect = self._connect


def _p10(values):
    values = sorted(values)
    return values[int(.1 * (len(values) - 1))] if values else None


def _version():
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=5).stdout.strip() or 'uncommitted'
    except OSError:
        return 'unknown'


def summarize(rows, pixels):
    ok = [r for r in rows if 'rubric' in r]
    dims = list(WEIGHTS)
    def block(items):
        out = {'cases': len(items), 'overall_mean': round(statistics.mean(r['rubric']['overall'] for r in items), 1) if items else None,
               'overall_p10': _p10([r['rubric']['overall'] for r in items])}
        for d in dims:
            vals = [r['rubric']['scores'][d] for r in items]
            out[d] = {'mean': round(statistics.mean(vals), 1) if vals else None, 'p10': _p10(vals),
                      'defect_rate': round(sum(any(x.split('.')[0] == d for x in r['rubric']['defects']) for r in items) / len(items), 3) if items else None}
        return out
    buckets = defaultdict(list)
    for r in ok:
        buckets[r['bucket']].append(r)
    times = sorted(r['seconds'] for r in rows); defects = defaultdict(int)
    for r in ok:
        for x in set(r['rubric']['defects']):
            defects[x] += 1
    return {
        'cases': len(rows), 'source_families': len({r['family'] for r in rows}),
        'valid': sum(r.get('valid', False) for r in rows), 'errors': sum('error' in r for r in rows),
        'tier0_failures': sum(bool(r.get('rubric', {}).get('tier0', ['error'])) for r in rows),
        'tier0_cases': sorted(r['case'] for r in rows if r.get('rubric', {}).get('tier0', ['error'])),
        'tier1_failures': dict(sorted(_count(x for r in ok for x in r['rubric']['tier1']).items())),
        'quality_passed': sum(not r['rubric']['defects'] for r in ok),
        'defect_counts': dict(sorted(defects.items(), key=lambda kv: -kv[1])),
        'tier2': block(ok), 'tier2_by_bucket': {b: block(v) for b, v in sorted(buckets.items())},
        'text_overflow_cases': sum(r.get('overflow', 0) > 0 for r in rows), 'collision_cases': sum(r.get('collisions', 0) > 0 for r in rows),
        'contrast_regression_cases': sum(r.get('contrast_regressions', 0) > 0 for r in rows),
        'render_failures': sum(r.get('rendered') is not True for r in rows) if pixels else None,
        'median_seconds': times[len(times) // 2] if times else 0, 'p95_seconds': times[min(len(times) - 1, int(.95 * len(times)))] if times else 0,
        'pixels': pixels, 'engine_version': _version(), 'platform': platform.platform(), 'python': platform.python_version(),
        'notes': 'Internal rubric; not the official scorer. Multiple targets share one source family.'}


def _count(items):
    out = defaultdict(int)
    for x in items:
        out[x] += 1
    return out


def run(corpus, folder, pixels=False, offline=False, formats=None, split=None, split_manifest=None, include_embedded=True):
    from training.families import split_of, resolve_family
    frozen_splits = json.loads(Path(split_manifest).read_text()) if split_manifest else None
    folder.mkdir(parents=True, exist_ok=True); rows = []; corpus_hash = hashlib.sha256()
    guard = _NoNetwork() if offline else None
    if guard:
        guard.__enter__()
    try:
        for path in sorted(corpus.glob('*.json')):
            if path.name == 'manifest.jsonl':
                continue
            raw = path.read_bytes(); task = json.loads(raw); source = task.get('source', task)
            family = resolve_family(task, path.stem)
            if split and split_of(family, frozen_splits) != split:
                continue
            corpus_hash.update(raw)
            targets = formats or FORMATS
            if include_embedded and 'target_canvas' in task and (task['target_canvas']['width'], task['target_canvas']['height']) not in targets:
                targets = [(task['target_canvas']['width'], task['target_canvas']['height'])] + list(targets)
            for w, h in targets:
                start = time.monotonic(); key = f'{path.stem}__{w}x{h}'; target = {'width': w, 'height': h}
                row = {'case': key, 'family': family, 'target': target, 'bucket': aspect_bucket(source['canvas'], target), 'direction': direction(source['canvas'], target)}
                try:
                    result = generate(source, target, top_k=1, options={'render': pixels})[0]; d = result['diagnostics']
                    rub = result['rubric']
                    if pixels:
                        from renderer.converter.convert import render_design_to_png
                        try:
                            render_design_to_png(result['design'], ASSETS_DIR, folder / (key + '.png'), max_size=800); row['preview'] = key + '.png'; row['rendered'] = True
                        except Exception as exc:
                            row['rendered'] = False; row['render_error'] = type(exc).__name__ + ': ' + str(exc)
                        if row['rendered'] is False:
                            from engine.rubric import evaluate
                            rub = evaluate(source, result['design'], None, d, result['score'], target, rendered=False)
                    row.update(valid=result['valid'] and rub['gate'], overflow=len(d['overflow']), collisions=len(d['collisions']),
                               contrast_regressions=len(d['contrast_regressions']), score=result['score']['total'], template=result['template'],
                               min_font=result['score']['stats']['min_font'], checks=result['checks'],
                               rubric={**rubric_summary(rub), 'scores': rub['tier2']['scores'], 'stats': rub['tier2']['stats'],
                                       'warnings': [x['id'] for x in rub['tier2']['warnings']]})
                    (folder / (key + '.json')).write_text(json.dumps({'source': source, 'target_canvas': target, 'result': result}, indent=2))
                except Exception as exc:
                    row.update(valid=False, error=type(exc).__name__ + ': ' + str(exc))
                row['seconds'] = round(time.monotonic() - start, 3); rows.append(row)
                print(key, 'PASS' if row['valid'] else 'FAIL', row.get('rubric', {}).get('overall', ''), ' '.join(row.get('rubric', {}).get('defects', [])), row.get('error', ''), flush=True)
    finally:
        if guard:
            guard.__exit__()
    summary = summarize(rows, pixels); summary['corpus_hash'] = corpus_hash.hexdigest()[:16]; summary['corpus'] = str(corpus); summary['split'] = split
    from evaluation.visual_suite import code_manifest, resource_manifest
    from engine.canonical import ROOT
    summary['code_snapshot'] = code_manifest(ROOT)
    summary['resource_snapshot'] = resource_manifest(ROOT)
    summary['evaluation_identity'] = {r['path']:r['sha256'] for r in summary['code_snapshot']['files'] if r['path'] in ('engine/rubric.py','engine/metrics.py','engine/judging.py','engine/quality.py','engine/invariants.py')}
    summary['offline_scope'] = 'Python socket guard only; native process network isolation not certified'
    summary['offline'] = offline; summary['network_attempts'] = guard.attempts if guard else None
    (folder / 'results.json').write_text(json.dumps({'summary': summary, 'cases': rows}, indent=2))
    (folder / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Layout evaluation</title><style>body{font:15px system-ui;background:#101918;color:#def;padding:24px}article{border:1px solid #456;padding:16px;margin:12px}img{max-width:800px;max-height:400px;background:white}pre{white-space:pre-wrap}</style><h1>Evaluation run</h1><pre id="report"></pre><script>fetch('results.json').then(r=>r.json()).then(d=>{document.getElementById('report').textContent=JSON.stringify(d.summary,null,2);for(const c of d.cases){let a=document.createElement('article');let h=document.createElement('h2');h.textContent=c.case;a.append(h);if(c.preview){let im=document.createElement('img');im.src=c.preview;a.append(im)}let p=document.createElement('pre');p.textContent=JSON.stringify({...c,checks:undefined},null,2);a.append(p);document.body.append(a)}})</script>''')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--corpus', type=Path, default=Path('input')); ap.add_argument('--name', default='samples')
    ap.add_argument('--render', action='store_true'); ap.add_argument('--offline', action='store_true', help='fail on any network connection (rubric D6)')
    ap.add_argument('--formats', help='comma list like 1080x1080,728x90 (default: full catalogue)')
    ap.add_argument('--split-manifest', type=Path); ap.add_argument('--exact-formats', action='store_true', help='ignore embedded target_canvas'); ap.add_argument('--split', choices=('train', 'validation', 'test'), help='keep only families in this training split (sha256(family) %% 100)')
    a = ap.parse_args()
    if not a.name.replace('-', '').replace('_', '').isalnum():
        ap.error('Use a simple run name')
    formats = [tuple(map(int, f.split('x'))) for f in a.formats.split(',')] if a.formats else None
    s = run(a.corpus, Path('output/evaluation') / a.name, a.render, a.offline, formats, a.split, a.split_manifest, not a.exact_formats)
    print(json.dumps({k: s[k] for k in ('cases', 'valid', 'tier0_failures', 'tier1_failures', 'quality_passed', 'defect_counts', 'median_seconds')} | {'overall': s['tier2']['overall_mean']}, indent=2))
