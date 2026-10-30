"""Build the blind study: 200 Crello designs x 5 target sizes; our engine, Gaurav's and the
merged engine (merged/), three blind options per task.

Run from the repo root (each stage is resumable and skips finished work):
    .venv/bin/python -m tools.blind_eval.build --gaurav ../chameleon-competitors/gaurav/chameleon

Stages: select designs and sizes -> run the engines -> render -> write pairs.json -> verify
that every published picture is the render of the layout its option names.
Add --publish to render and publish only the finished tasks, so judging can start early.
The design pool comes from `tools.build_crello_catalog --output work/blind-eval/crello`.
"""
from __future__ import annotations

import argparse
import hashlib
import random
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
ENGINES = ('pranay', 'gaurav', 'merged')   # run by worker.py, compared by the judges
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
        if name not in roots:
            continue
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


def fingerprint(design):
    """Hash of a layout, stored beside its render so a stale picture is never reused."""
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:16]


def render_one(job):
    from renderer.converter.convert import render_design_to_png
    design, png = job
    try:
        render_design_to_png(design, ROOT / 'brand_kit/assets', png, max_size=900)
        png.with_suffix('.layout').write_text(fingerprint(design))
        return None
    except Exception as exc:  # a broken output is shown as "no output", never hidden
        png.unlink(missing_ok=True)
        return f'{png.name}: {exc}'


def render(tasks):
    """Render every output whose picture is missing or was drawn from a different layout."""
    img = WORK / 'img'
    img.mkdir(exist_ok=True)
    jobs = [(source_of(s), img / f'{s}__source.png') for s in {t['sample'] for t in tasks}]
    for name in ENGINES:
        for t in tasks:
            rec = json.loads((WORK / 'runs' / name / f"{t['id']}.json").read_text())
            png = img / f"{t['id']}__{name}.png"
            if rec['ok']:
                jobs.append((rec['design'], png))
            else:  # an engine that now refuses must not keep showing an old picture
                png.unlink(missing_ok=True)
    def current(design, png):
        stamp = png.with_suffix('.layout')
        return png.exists() and stamp.exists() and stamp.read_text() == fingerprint(design)
    jobs = [j for j in jobs if not current(*j)]
    with ProcessPoolExecutor(8) as pool:
        errors = [e for e in pool.map(render_one, jobs, chunksize=8) if e]
    print(f'Rendered {len(jobs) - len(errors)} images, {len(errors)} failed', *errors[:10], sep='\n')


def pairs(tasks):
    """Shuffle each task's options by a seeded draw so names never leak through position.

    Each option records the fingerprint of its layout and of its picture: the judge serves
    pictures under a URL that includes the picture fingerprint, so a browser can never show
    a cached picture from an earlier build under a new option letter.
    """
    out = []
    for t in tasks:
        order = list(ENGINES)
        random.Random(digest(t['id'], 'order', *ENGINES)).shuffle(order)
        meta = {}
        for name in ENGINES:
            rec = json.loads((WORK / 'runs' / name / f"{t['id']}.json").read_text())
            png = WORK / 'img' / f"{t['id']}__{name}.png"
            ok = rec['ok'] and png.exists()
            meta[name] = {'ok': ok, 'seconds': rec['seconds'],
                          'layout': fingerprint(rec['design']) if rec['ok'] else None,
                          'render': hashlib.sha256(png.read_bytes()).hexdigest()[:16] if ok else None}
        out.append({**t, 'options': dict(zip('ABC', order)), 'engines': meta})
    tmp = WORK / 'pairs.tmp'  # the judge app may be reading the previous file
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(WORK / 'pairs.json')
    fails = {n: sum(not p['engines'][n]['ok'] for p in out) for n in ENGINES}
    print(f'{len(out)} tasks written; failed outputs per engine: {fails}')


def verify(sample=40):
    """Check every published option against the data it claims to show. Exits non-zero on any
    mismatch: a wrong picture under a name would silently corrupt every vote on it."""
    tasks = json.loads((WORK / 'pairs.json').read_text())
    problems = []
    for t in tasks:
        if sorted(t['options'].values()) != sorted(ENGINES):
            problems.append(f"{t['id']}: options {t['options']}")
        source = source_of(t['sample'])
        for name, meta in t['engines'].items():
            rec = json.loads((WORK / 'runs' / name / f"{t['id']}.json").read_text())
            png = WORK / 'img' / f"{t['id']}__{name}.png"
            if meta['ok'] != (rec['ok'] and png.exists()):
                problems.append(f"{t['id']}/{name}: ok flag")
            if not meta['ok']:
                continue
            d = rec['design']
            if fingerprint(d) != meta['layout'] or png.with_suffix('.layout').read_text() != meta['layout']:
                problems.append(f"{t['id']}/{name}: picture not drawn from this layout")
            if hashlib.sha256(png.read_bytes()).hexdigest()[:16] != meta['render']:
                problems.append(f"{t['id']}/{name}: picture changed since publishing")
            if d['canvas'] != t['target_canvas'] or sorted(e['id'] for e in d['elements']) != sorted(e['id'] for e in source['elements']):
                problems.append(f"{t['id']}/{name}: canvas or element ids differ from the task")
    # Spot-check that a fresh render of the layout reproduces the published picture.
    import tempfile
    from PIL import Image, ImageChops
    from renderer.converter.convert import render_design_to_png
    rng = random.Random(digest('verify'))
    checks = [(t, n) for t in rng.sample(tasks, min(sample, len(tasks))) for n in ENGINES if t['engines'][n]['ok']]
    with tempfile.TemporaryDirectory() as tmp:
        for t, name in checks:
            design = json.loads((WORK / 'runs' / name / f"{t['id']}.json").read_text())['design']
            fresh = Path(tmp) / 'fresh.png'
            render_design_to_png(design, ROOT / 'brand_kit/assets', fresh, max_size=900)
            a = Image.open(fresh).convert('RGB')
            b = Image.open(WORK / 'img' / f"{t['id']}__{name}.png").convert('RGB')
            if a.size != b.size or ImageChops.difference(a, b).getbbox() is not None:
                problems.append(f"{t['id']}/{name}: re-render differs from published picture")
    print(f'Verified {len(tasks)} tasks x {len(ENGINES)} engines; re-rendered {len(checks)} pictures; '
          f'{len(problems)} problems', *problems[:20], sep='\n')
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gaurav', type=Path, required=True, help="the competitor's chameleon/ folder")
    ap.add_argument('--pranay', type=Path, default=WORK / 'engines/pranay', help='our engine snapshot')
    ap.add_argument('--count', type=int, default=200)
    ap.add_argument('--shards-pranay', type=int, default=4)
    ap.add_argument('--shards-gaurav', type=int, default=2)
    ap.add_argument('--shards-merged', type=int, default=5)
    ap.add_argument('--publish', action='store_true',
                    help='only render and publish the tasks every engine has finished (while runs continue)')
    a = ap.parse_args()
    tasks = select(a.count)
    if a.publish:
        tasks = [t for t in tasks if all((WORK / 'runs' / n / f"{t['id']}.json").exists() for n in ENGINES)]
    else:
        # The merged engine runs on our engine snapshot, with merged/ from this repository.
        run(tasks, {'pranay': a.pranay.resolve(), 'gaurav': a.gaurav.resolve(), 'merged': a.pranay.resolve()},
            {'pranay': a.shards_pranay, 'gaurav': a.shards_gaurav, 'merged': a.shards_merged})
    render(tasks)
    pairs(tasks)
    verify()
