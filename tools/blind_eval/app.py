"""Blind judging of our engine, Gaurav's and the merged engine by a panel of three.

Build the study first with `tools.blind_eval.build`, then serve it on the local network:
    .venv/bin/uvicorn tools.blind_eval.app:app --host 0.0.0.0 --port 8790

Engine names never reach the browser unless a viewer turns blind mode off or opens the
results, and each task's A/B/C order is fixed by a seeded shuffle in pairs.json. When two
engines produce the same picture, a vote for it counts for both.

Pictures are served under URLs that carry their own fingerprint, and votes are stored by the
engines they credit, so neither a browser cache nor a rebuilt study can attach a picture or
a past vote to the wrong option.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get('BLIND_EVAL_DIR') or ROOT / 'work/blind-eval')
HERE = Path(__file__).parent
JUDGES = ('pranay', 'gaurav', 'kavya')
ENGINES = ('pranay', 'gaurav', 'merged')
OPTIONS = ('A', 'B', 'C')
CHOICES = (*OPTIONS, 'none', 'all-bad')

app = FastAPI(title='Chameleon Blind Judge', version='1.0')
_lock = threading.Lock()


def load_pairs():
    path = WORK / 'pairs.json'
    if not path.is_file():
        raise HTTPException(503, 'The study is not built yet. Run tools.blind_eval.build first.')
    return json.loads(path.read_text())


PAIRS: list[dict] = []
BY_ID: dict[str, dict] = {}
SAMPLES: dict[str, dict] = {}
PLANNED: list[str] = []
_loaded = {'mtime': None}


def pairs():
    """The published pairs, reloaded whenever the build publishes more of them."""
    path = WORK / 'pairs.json'
    mtime = path.stat().st_mtime if path.is_file() else None
    if mtime != _loaded['mtime']:
        fresh = load_pairs()
        with _lock:
            PAIRS[:] = fresh
            BY_ID.clear()
            BY_ID.update({p['id']: p for p in fresh})
            SAMPLES.update({s['sample']: s for s in json.loads((WORK / 'samples.json').read_text())})
            PLANNED[:] = [t['id'] for t in json.loads((WORK / 'tasks.json').read_text())]
            _loaded['mtime'] = mtime
    return PAIRS


def db():
    con = sqlite3.connect(WORK / 'votes.sqlite')
    con.execute('''create table if not exists votes (judge text, pair text, choice text, winner text,
                   note text, seconds real, at real, blind integer, primary key (judge, pair))''')
    return con


def core_tasks():
    """The core set (core.json), judged before everything else; empty until it is chosen."""
    path = WORK / 'core.json'
    return json.loads(path.read_text())['tasks'] if path.is_file() else []


def shared_order():
    """One shuffled order for the panel, so sizes and designs do not come in runs; the core
    set comes first, then every other task."""
    pairs()
    ids = list(PLANNED)  # shuffle the full plan, so tasks published later keep the order stable
    random.Random(hashlib.sha256(b'order|panel').hexdigest()).shuffle(ids)
    first = set(core_tasks())
    ids = [i for i in ids if i in first] + [i for i in ids if i not in first]
    return [i for i in ids if i in BY_ID]


def votes_by_pair(con, pair_id=None):
    out = defaultdict(dict)
    rows = con.execute('select pair, judge, choice from votes' + (' where pair=?' if pair_id else ''),
                       (pair_id,) if pair_id else ())
    for pid, judge, choice in rows:
        out[pid][judge] = choice
    return out


def winner_of(pair, choice):
    """The engines a vote credits: every engine whose render is the picked picture."""
    if choice not in OPTIONS:
        return choice
    picked = pair['engines'][pair['options'][choice]]['render']
    return ','.join(e for e in ENGINES if picked and pair['engines'][e]['render'] == picked)


def letter_of(pair, winner, choice=None):
    """The option now showing the picture a stored vote picked (votes outlive option order).

    When two options show the same picture, the letter the judge pressed is kept if it
    still shows that picture.
    """
    if winner in ('none', 'all-bad'):
        return winner
    names = winner.split(',')
    if choice in OPTIONS and pair['options'][choice] in names:
        return choice
    return next((side for side in OPTIONS if pair['options'][side] in names), None)


def check_judge(judge):
    if judge not in JUDGES:
        raise HTTPException(404, 'Unknown judge')


@app.get('/')
def index():
    return FileResponse(HERE / 'index.html', headers={'Cache-Control': 'no-store'})


@app.get('/api/study')
def study():
    with db() as con:
        done = dict(con.execute('select judge, count(*) from votes group by judge').fetchall())
    return {'judges': JUDGES, 'total': len(pairs()), 'planned': len(PLANNED), 'samples': len({p['sample'] for p in pairs()}),
            'done': {j: done.get(j, 0) for j in JUDGES}}


@app.get('/api/queue')
def queue():
    """The panel's order and every judge's votes so far."""
    with db() as con:
        votes = votes_by_pair(con)
    return {'order': shared_order(), 'votes': votes, 'planned': len(PLANNED), 'core': core_tasks()}


@app.get('/api/pair/{pair_id}')
def pair(pair_id: str, reveal: bool = False):
    pairs()
    p = BY_ID.get(pair_id)
    if p is None:
        raise HTTPException(404, 'Pair not found')
    s = SAMPLES[p['sample']]
    with db() as con:
        votes = {j: letter_of(p, w, c) for j, w, c in con.execute('select judge, winner, choice from votes where pair=?', (pair_id,))}
        note = con.execute("select note from votes where pair=? and note != '' order by at desc", (pair_id,)).fetchone()
    def picture(side):
        meta = p['engines'][p['options'][side]]
        # Identical pictures are flagged by letter (never by engine), so judges know a vote
        # for either counts for both.
        same = next((o for o in OPTIONS if o < side and meta['ok'] and p['engines'][p['options'][o]]['render'] == meta['render']), None)
        return {'ok': meta['ok'], 'img': f"/img/{pair_id}/{side}?v={meta['render']}", 'same': same}
    out = {'id': pair_id, 'title': s['title'], 'source_size': [s['width'], s['height']],
           'target': {'label': p['label'], **p['target_canvas']},
           'sides': {side: picture(side) for side in OPTIONS},
           'source_img': f"/img/{pair_id}/source?v={p['sample']}", 'votes': votes, 'note': note[0] if note else ''}
    if reveal:
        for side in OPTIONS:
            out['sides'][side]['engine'] = p['options'][side]
    return out


@app.get('/img/{pair_id}/{which}')
def image(pair_id: str, which: Literal['A', 'B', 'C', 'source'], v: str = ''):
    pairs()
    p = BY_ID.get(pair_id)
    if p is None:
        raise HTTPException(404, 'Pair not found')
    if which == 'source':
        path, expected = WORK / 'img' / f"{p['sample']}__source.png", p['sample']
    else:
        path, expected = WORK / 'img' / f"{pair_id}__{p['options'][which]}.png", p['engines'][p['options'][which]]['render']
    if v != expected:  # an old page asking for a picture this option no longer shows
        raise HTTPException(409, 'This picture has changed; reload the page')
    if not path.is_file():
        raise HTTPException(404, 'No output')
    if which != 'source' and hashlib.sha256(path.read_bytes()).hexdigest()[:16] != expected:
        raise HTTPException(409, 'Picture differs from the published study; rebuild it')
    # The file name contains the engine, so it is never sent: only the bytes are. The URL
    # carries the picture's fingerprint, so caching it can never show the wrong picture.
    return Response(path.read_bytes(), media_type='image/png', headers={'Cache-Control': 'private, max-age=86400, immutable'})


class Vote(BaseModel):
    judge: str
    pair: str
    choice: Literal['A', 'B', 'C', 'none', 'all-bad']
    note: str = Field(default='', max_length=500)
    seconds: float = Field(default=0, ge=0, le=86400)
    blind: bool = True


@app.post('/api/vote')
def vote(v: Vote):
    check_judge(v.judge)
    pairs()
    p = BY_ID.get(v.pair)
    if p is None:
        raise HTTPException(404, 'Pair not found')
    with _lock, db() as con:
        con.execute('insert or replace into votes values (?,?,?,?,?,?,?,?)',
                    (v.judge, v.pair, v.choice, winner_of(p, v.choice), v.note.strip(), v.seconds, time.time(), int(v.blind)))
    return {'ok': True}


@app.delete('/api/vote/{judge}/{pair_id}')
def unvote(judge: str, pair_id: str):
    check_judge(judge)
    with _lock, db() as con:
        con.execute('delete from votes where judge=? and pair=?', (judge, pair_id))
    return {'ok': True}


def fleiss_kappa(rows, categories):
    """Fleiss' kappa over items rated by the same number (>= 2) of raters."""
    rows = [r for r in rows if len(r) >= 2]
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    n = len(rows[0])
    counts = [Counter(r) for r in rows]
    p_i = [(sum(c[k] ** 2 for k in categories) - n) / (n * (n - 1)) for c in counts]
    p_j = [sum(c[k] for c in counts) / (len(rows) * n) for k in categories]
    pe = sum(x * x for x in p_j)
    return None if pe == 1 else round((sum(p_i) / len(rows) - pe) / (1 - pe), 3)


def tally(winners):
    """Wins per engine (a shared picture credits each engine that made it), plus no-pick votes."""
    c = Counter()
    for w in winners:
        c.update(w.split(',') if w not in ('none', 'all-bad') else [w])
    picked = sum(w not in ('none', 'all-bad') for w in winners)
    return {**{k: c[k] for k in (*ENGINES, 'none', 'all-bad')}, 'n': len(winners),
            **{f'{e}_share': round(c[e] / picked, 3) if picked else None for e in ENGINES}}


def verdict_of(winners):
    """The 2-of-3 verdict: engines picked by a majority, else the majority's no-pick, else split."""
    backed = [e for e in ENGINES if sum(e in w.split(',') for w in winners) >= 2]
    if backed:
        return ','.join(backed)
    if len(winners) < len(JUDGES):
        return 'pending'
    top, n = Counter(winners).most_common(1)[0]
    return top if n >= 2 and top in ('none', 'all-bad') else 'split'


@app.get('/api/results')
def results(scope: Literal['all', 'core'] = 'all'):
    """Tallies over every task, or (scope=core) over the core set only."""
    by_pair = defaultdict(dict)
    pairs()
    keep = set(core_tasks()) if scope == 'core' else set(BY_ID)
    with db() as con:
        for judge, pair_id, winner in con.execute('select judge, pair, winner from votes'):
            if pair_id in keep:
                by_pair[pair_id][judge] = winner
    per_judge = {j: tally([v[j] for v in by_pair.values() if j in v]) for j in JUDGES}
    majority, by_size, by_orient, rows = Counter(), defaultdict(list), defaultdict(list), []
    for pair_id, v in by_pair.items():
        p, s = BY_ID[pair_id], SAMPLES[BY_ID[pair_id]['sample']]
        verdict = verdict_of(list(v.values()))
        if verdict != 'pending':
            majority[verdict] += 1
        ratio = s['width'] / s['height']
        orient = 'square' if .95 <= ratio <= 1.05 else 'landscape' if ratio > 1 else 'portrait'
        for w in v.values():
            by_size[p['label']].append(w)
            by_orient[orient].append(w)
        rows.append({'pair': pair_id, 'title': s['title'], 'target': p['label'], **p['options'],
                     **{j: v.get(j) for j in JUDGES}, 'verdict': verdict})
    # Agreement is on the picture picked, so identical pictures under two letters agree.
    complete = [list(v.values()) for v in by_pair.values() if len(v) == len(JUDGES)]
    tasks = [p for p in PAIRS if p['id'] in keep]
    failures = {e: sum(not p['engines'][e]['ok'] for p in tasks) for e in ENGINES}
    seconds = {e: round(sorted(p['engines'][e]['seconds'] for p in tasks)[len(tasks) // 2], 2) if tasks else None for e in ENGINES}
    return {'scope': scope, 'total_pairs': len(tasks), 'fully_judged': len(complete),
            'overall': tally([w for v in by_pair.values() for w in v.values()]),
            'per_judge': per_judge, 'majority': dict(majority),
            'kappa': fleiss_kappa(complete, sorted({w for r in complete for w in r})),
            'by_size': {k: tally(v) for k, v in sorted(by_size.items())},
            'by_orientation': {k: tally(v) for k, v in sorted(by_orient.items())},
            'engine_failures': failures, 'median_seconds': seconds, 'rows': rows}


@app.get('/api/export.csv')
def export():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['judge', 'pair', 'sample', 'target', 'A', 'B', 'C', 'choice', 'winner', 'note', 'seconds', 'at', 'blind'])
    pairs()
    with db() as con:
        for judge, pair_id, choice, winner, note, secs, at, blind in con.execute('select * from votes order by at'):
            p = BY_ID[pair_id]
            o = p['options']
            w.writerow([judge, pair_id, p['sample'], p['label'], o['A'], o['B'], o['C'], letter_of(p, winner, choice), winner, note, secs, at, blind])
    return Response(buf.getvalue(), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="blind-votes.csv"'})
