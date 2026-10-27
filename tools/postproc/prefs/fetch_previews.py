"""Fetch Crello's own rendered previews (true designer renders) for the preference-round designs."""
import json, pickle, sys, time
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from training.retarget.fetch_crello import REPO, REVISION
from training.retarget.data import load

OUT = Path('work/postproc/prefs/previews'); OUT.mkdir(parents=True, exist_ok=True)
designs, pairs = load('work/retarget-data')
rounds = json.load(open('work/postproc/prefs/rounds.json'))
want = set()
for pis in rounds.values():
    for pi in pis:
        want |= {designs[pairs[pi]['src']]['id'], designs[pairs[pi]['tgt']]['id']}
meta = json.load(open('work/retarget-scale/rg_meta.json'))
raw = {f.name: pq.read_table(f, columns=['id'])['id'].to_pylist() for f in sorted(Path('work/retarget-data/raw').glob('*.parquet'))}
tasks, pos = [], {}
for m in meta:
    s = pos.get(m['shard'], 0); ids = raw[m['shard']][s:s + m['rows']]; pos[m['shard']] = s + m['rows']
    hit = want & set(ids)
    if hit:
        tasks.append((m['shard'], m['rg'], hit))
print(len(want), 'designs in', len(tasks), 'row groups', flush=True)
fs = HfFileSystem()
for shard, rg, hit in tasks:
    if all((OUT / f'{i}.png').exists() for i in hit):
        continue
    for attempt in range(5):
        try:
            with fs.open(f"{REPO}/data/{shard}", revision=REVISION, block_size=1 << 22) as fh:
                tb = pq.ParquetFile(fh).read_row_group(rg, columns=['id', 'preview'])
            for row in tb.to_pylist():
                if row['id'] in hit and row['preview']:
                    (OUT / f"{row['id']}.png").write_bytes(row['preview']['bytes'])
            print(shard, rg, len(hit), flush=True)
            break
        except Exception as e:
            print('retry', repr(e)[:80], flush=True); time.sleep(5 * (attempt + 1))
