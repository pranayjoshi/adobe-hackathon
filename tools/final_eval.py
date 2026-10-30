"""Test the final engine on designs nothing was fitted on, as a blind study of its own.

    .venv/bin/python -m tools.final_eval

Takes the catalogue's designs left out of the 200-design study, two sizes each, and runs the
final engine, Gaurav's, the merged engine, and the final engine without Gaurav's layouts. Writes a
judge-ready folder at work/final-eval (BLIND_EVAL_DIR=work/final-eval
BLIND_EVAL_ENGINES=final,gaurav,merged) and report.json with the structural results.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.blind_eval.build as b  # noqa: E402

STUDY = b.WORK
OUT = ROOT / 'work/final-eval'
RUN = ('final', 'gaurav', 'merged', 'final_solo')
JUDGED = ('final', 'gaurav', 'merged')
GAURAV = ROOT.parent / 'chameleon-competitors/gaurav/chameleon'


def held_out():
    catalog = json.loads((STUDY / 'crello/catalog.json').read_text())
    ready = [e for e in catalog['entries'] if e['status'] == 'ready' and e['text_elements'] >= 1 and e['elements'] >= 2]
    ready.sort(key=lambda e: b.digest(e['id']))
    usage, samples, tasks = Counter(), [], []
    for done, e in enumerate(ready[200:]):
        sizes = b.pick_sizes({'width': e['width'], 'height': e['height']}, usage, done, count=2)
        samples.append({'sample': e['id'], 'title': e['title'], 'format': e['format'], 'width': e['width'],
                        'height': e['height'], 'sizes': [p['id'] for p in sizes]})
        tasks += [{'id': f"{e['id']}__{p['id']}", 'sample': e['id'], 'preset': p['id'], 'label': p['label'],
                   'target_canvas': {'width': p['width'], 'height': p['height']}} for p in sizes]
    (OUT / 'samples.json').write_text(json.dumps(samples, indent=1))
    (OUT / 'tasks.json').write_text(json.dumps(tasks, indent=1))
    return tasks


def report(tasks):
    from engine.invariants import validate
    from engine.pipeline import analyze, assess_design
    from final.pixels import features as pixel_features
    from final.prefs import layout_features, score
    rows = {n: Counter() for n in RUN}
    scores = {}
    for t in tasks:
        source = b.source_of(t['sample'])
        for n in RUN:
            rec = json.loads((OUT / 'runs' / n / f"{t['id']}.json").read_text())
            rows[n]['tasks'] += 1
            if not rec.get('ok'):
                rows[n]['no_output'] += 1
                continue
            rows[n]['seconds'] += rec['seconds']
            bad = [c['name'] for c in validate(source, rec['design']) if not c['passed']]
            rows[n]['structurally_clean'] += not bad
            for c in bad:
                rows[n]['fail: ' + c] += 1
            r = assess_design(source, t['target_canvas'], rec['design'], analyze(source), n)
            scores[t['id'], n] = score(layout_features(source, r) + pixel_features(rec['design']))
    wins = Counter()
    for t in tasks:
        have = {n: scores[t['id'], n] for n in JUDGED if (t['id'], n) in scores}
        wins[max(have, key=have.get)] += 1
    result = {'engines': rows, 'judge_model_picks': wins, 'note': 'judge_model_picks is the fitted model, not people'}
    (OUT / 'report.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'runs').mkdir(exist_ok=True)
    if not (OUT / 'crello').exists():
        (OUT / 'crello').symlink_to(STUDY / 'crello')
    tasks = held_out()
    b.WORK, b.ENGINES = OUT, RUN
    print(f'{len(tasks)} tasks', flush=True)
    snapshot = (STUDY / 'engines/pranay').resolve()
    b.run(tasks, {'pranay': snapshot, 'final': ROOT, 'gaurav': GAURAV.resolve(), 'merged': snapshot, 'final_solo': ROOT},
          {'final': 4, 'gaurav': 1, 'merged': 2, 'final_solo': 3})
    b.ENGINES = JUDGED
    b.render(tasks)
    b.pairs(tasks)
    b.verify()
    report(tasks)


if __name__ == '__main__':
    main()
