"""Build the blind A/B study: 200 Crello designs x 5 target sizes, both engines.

Run from the repo root (each stage is resumable and skips finished work):
    .venv/bin/python -m tools.blind_eval.build --gaurav ../chameleon-competitors/gaurav/chameleon

Stages: select designs and sizes -> run both engines -> render -> write pairs.json.
Add --publish to render and publish only the finished pairs, so judging can start early.
The design pool comes from `tools.build_crello_catalog --output work/blind-eval/crello`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lite.app import PRESETS, SAME_SHAPE  # noqa: E402  the 14 sizes the apps offer

WORK = ROOT / 'work/blind-eval'
ENGINES = ('pranay', 'gaurav')
SEED = 'chameleon-blind-v1'


def digest(*parts):
    return hashlib.sha256('|'.join([SEED, *map(str, parts)]).encode()).hexdigest()


def shape(width, height):
    """A size as a point: log aspect dominates, log area separates same-shaped sizes."""
    return math.log(width / height), 0.35 * math.log(width * height / 1e6)


def pick_sizes(canvas, usage, done, count=5):
    """The `count` presets most unlike the source and each other (greedy farthest point).

    Same-shape targets are skipped: they are a plain scale, not a design decision. Distances
    shrink for presets already used often across the `done` earlier designs, so all 14 sizes
    appear in the study instead of only the two most extreme ad formats.
    """
    a = canvas['width'] / canvas['height']
    pool = [p for p in PRESETS if abs((p['width'] / p['height']) / a - 1) > SAME_SHAPE]
    chosen, anchors = [], [shape(canvas['width'], canvas['height'])]
    while len(chosen) < count and pool:
        def score(p):
            here = shape(p['width'], p['height'])
            return min(math.dist(here, x) for x in anchors) / (1 + 3 * usage[p['id']] / max(1, done))
        best = max(pool, key=lambda p: (score(p), p['id']))
        pool.remove(best)
        chosen.append(best)
        anchors.append(shape(best['width'], best['height']))
        usage[best['id']] += 1
    return chosen


def select(count):
    catalog = json.loads((WORK / 'crello/catalog.json').read_text())
    ready = [e for e in catalog['entries'] if e['status'] == 'ready' and e['text_elements'] >= 1 and e['elements'] >= 2]
    ready.sort(key=lambda e: digest(e['id']))
    if len(ready) < count:
        raise SystemExit(f'Only {len(ready)} ready designs; import more with tools.build_crello_catalog')
    usage, tasks, samples = Counter(), [], []
    for done, e in enumerate(ready[:count]):
        sizes = pick_sizes({'width': e['width'], 'height': e['height']}, usage, done)
        samples.append({'sample': e['id'], 'title': e['title'], 'format': e['format'],
                        'width': e['width'], 'height': e['height'], 'sizes': [p['id'] for p in sizes]})
        for p in sizes:
            tasks.append({'id': f"{e['id']}__{p['id']}", 'sample': e['id'], 'preset': p['id'],
                          'label': p['label'], 'target_canvas': {'width': p['width'], 'height': p['height']}})
    (WORK / 'samples.json').write_text(json.dumps(samples, indent=1))
    (WORK / 'tasks.json').write_text(json.dumps(tasks, indent=1))
    print(f'{len(samples)} designs, {len(tasks)} tasks. Size usage: {dict(usage.most_common())}')
    return tasks


def source_of(sample):
    return json.loads((WORK / 'crello/designs' / f'{sample}.json').read_text())['source']


def run(tasks, roots, shards):
    procs, renderer_root = [], roots['pranay']
    for name in ENGINES:
        out = WORK / 'runs' / name
        out.mkdir(parents=True, exist_ok=True)
        todo = [t for t in tasks if not (out / f"{t['id']}.json").exists()]
        for i in range(shards[name]):
            part = [{**t, 'source': source_of(t['sample'])} for t in todo[i::shards[name]]]
            if not part:
                continue
            shard = WORK / 'shards' / f'{name}-{i}.json'
            shard.parent.mkdir(exist_ok=True)
            shard.write_text(json.dumps(part))
            log = open(WORK / 'shards' / f'{name}-{i}.log', 'a')
            procs.append(subprocess.Popen([str(ROOT / '.venv/bin/python'), str(Path(__file__).with_name('worker.py')),
                                           name, str(roots[name]), str(renderer_root), str(shard), str(out)],
                                          cwd=roots[name], stdout=log, stderr=subprocess.STDOUT))
    print(f'{len(procs)} workers running; progress in {WORK / "shards"}/*.log', flush=True)
    for p in procs:
        p.wait()


def render_one(job):
    from renderer.converter.convert import render_design_to_png
    design, png = job
    try:
        render_design_to_png(design, ROOT / 'brand_kit/assets', png, max_size=900)
        return None
    except Exception as exc:  # a broken output is shown as "no output", never hidden
        return f'{png.name}: {exc}'


def render(tasks):
    img = WORK / 'img'
    img.mkdir(exist_ok=True)
    jobs = [(source_of(s), img / f'{s}__source.png') for s in {t['sample'] for t in tasks}]
    for name in ENGINES:
        for t in tasks:
            rec = json.loads((WORK / 'runs' / name / f"{t['id']}.json").read_text())
            if rec['ok']:
                jobs.append((rec['design'], img / f"{t['id']}__{name}.png"))
    jobs = [j for j in jobs if not j[1].exists()]
    with ProcessPoolExecutor(8) as pool:
        errors = [e for e in pool.map(render_one, jobs, chunksize=8) if e]
    print(f'Rendered {len(jobs) - len(errors)} images, {len(errors)} failed', *errors[:10], sep='\n')


def pairs(tasks):
    """Fix each task's left/right order by a seeded coin so names never leak through position."""
    out = []
    for t in tasks:
        left, right = ENGINES if int(digest(t['id'], 'side'), 16) % 2 == 0 else ENGINES[::-1]
        meta = {}
        for name in ENGINES:
            rec = json.loads((WORK / 'runs' / name / f"{t['id']}.json").read_text())
            png = WORK / 'img' / f"{t['id']}__{name}.png"
            meta[name] = {'ok': rec['ok'] and png.exists(), 'seconds': rec['seconds']}
        out.append({**t, 'A': left, 'B': right, 'engines': meta})
    tmp = WORK / 'pairs.tmp'  # the judge app may be reading the previous file
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(WORK / 'pairs.json')
    fails = {n: sum(not p['engines'][n]['ok'] for p in out) for n in ENGINES}
    print(f'{len(out)} pairs written; failed outputs per engine: {fails}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gaurav', type=Path, required=True, help="the competitor's chameleon/ folder")
    ap.add_argument('--pranay', type=Path, default=WORK / 'engines/pranay', help='our engine snapshot')
    ap.add_argument('--count', type=int, default=200)
    ap.add_argument('--shards-pranay', type=int, default=4)
    ap.add_argument('--shards-gaurav', type=int, default=2)
    ap.add_argument('--publish', action='store_true',
                    help='only render and publish the pairs both engines have finished (while runs continue)')
    a = ap.parse_args()
    tasks = select(a.count)
    if a.publish:
        done = [t for t in tasks if all((WORK / 'runs' / n / f"{t['id']}.json").exists() for n in ENGINES)]
        render(done)
        pairs(done)
        raise SystemExit
    run(tasks, {'pranay': a.pranay.resolve(), 'gaurav': a.gaurav.resolve()},
        {'pranay': a.shards_pranay, 'gaurav': a.shards_gaurav})
    render(tasks)
    pairs(tasks)
