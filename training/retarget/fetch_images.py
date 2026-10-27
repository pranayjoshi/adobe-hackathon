"""Fetch layer images + previews for a small, held-out visual-check subset of Crello designs.

Reads only the selected row groups (see build of rowgroups.json) and keeps only the wanted rows.
"""
import argparse
import json
import pickle
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from training.retarget.fetch_crello import REPO, REVISION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='work/retarget-data/rowgroups.json')
    ap.add_argument('--families', type=int, default=60)
    ap.add_argument('--out', default='work/retarget-data/images')
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    plan.sort(key=lambda r: -len(r['fams']) / max(r['mb'], 1))
    got, picked = set(), []
    for r in plan:
        if len(set(r['fams']) - got) >= 3:
            got |= set(r['fams']); picked.append(r)
        if len(got) >= a.families:
            break
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for r in picked:
        want = set(r['hits'])
        if all((out / f'{i}.pkl').exists() for i in want):
            continue
        with fs.open(f"{REPO}/data/{r['shard']}", revision=REVISION, block_size=1 << 22) as fh:
            t = pq.ParquetFile(fh).read_row_group(r['rg'], columns=['id', 'image', 'preview'])
        for row in t.to_pylist():
            if row['id'] in want:
                rec = {'preview': (row['preview'] or {}).get('bytes'),
                       'layers': [(im or {}).get('bytes') for im in row['image']]}
                with open(out / f"{row['id']}.pkl", 'wb') as fh:
                    pickle.dump(rec, fh)
        print(r['shard'], r['rg'], len(want), flush=True)


if __name__ == '__main__':
    main()
