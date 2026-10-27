"""Fetch only the layout columns of cyberagent/crello (no layer images or previews).

Parquet range reads pull ~0.5 MB per shard instead of ~500 MB. Output: one parquet per shard.
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = 'datasets/cyberagent/crello'
REVISION = '5.1.0'
COLUMNS = ['id', 'length', 'group', 'format', 'canvas_width', 'canvas_height', 'category', 'title',
           'type', 'left', 'top', 'width', 'height', 'angle', 'opacity', 'color', 'text', 'font',
           'font_size', 'font_bold', 'font_italic', 'text_line', 'text_color', 'text_align',
           'capitalize', 'line_height', 'letter_spacing', 'cluster_index']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='work/retarget-data/raw')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for path in sorted(fs.ls(f'{REPO}/data', detail=False, revision=REVISION)):
        dest = out / Path(path).name
        if dest.exists():
            continue
        with fs.open(path, revision=REVISION, block_size=1 << 20) as fh:
            table = pq.ParquetFile(fh).read(columns=COLUMNS)
        pq.write_table(table, dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').rename(dest)
        print(dest.name, table.num_rows, flush=True)


if __name__ == '__main__':
    main()
