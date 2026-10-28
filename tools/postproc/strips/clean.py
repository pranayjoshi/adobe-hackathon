"""Keep synthetic strips that are good teaching examples: every text at least 15% above its role's
readable floor (targets that hug the floor teach the adapter fonts that sampling pushes under it)
and an appeal score of at least 0.1.

  python tools/postproc/strips/clean.py work/postproc/strips/synthetic.jsonl work/postproc/strips/synthetic_clean.jsonl
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness  # noqa: E402

MARGIN, MIN_APPEAL = 1.15, 0.1


def main(src_file, out_file):
    warnings.filterwarnings('ignore')
    harness.init('methods:current')
    from engine.blocks import FLOORS
    E = harness.ENV
    kept = total = 0
    seen = set()
    with open(out_file, 'w') as fh:
        for line in open(src_file):
            r = json.loads(line)
            if not r.get('kept') or not r.get('target') or (r['src'], r['fmt']) in seen:
                continue
            seen.add((r['src'], r['fmt']))
            total += 1
            _, A, _ = E.source(r['src'])
            ratios = [e['font_size'] / FLOORS.get(A.roles.get(f"el_{e['i']}"), 9)
                      for e in r['target']['els'] if e['type'] == 1 and e.get('font_size')]
            if r.get('appeal', 0) >= MIN_APPEAL and (not ratios or min(ratios) >= MARGIN):
                fh.write(line); kept += 1
    print(json.dumps({'kept': kept, 'of': total}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
