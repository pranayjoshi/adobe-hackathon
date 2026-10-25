"""Durable local workspace repository. Resolve root per request, never globally.

JSON files remain readable exports; SQLite owns job claims, ordered events,
revision identity and idempotent feedback. Transactions contain no model work.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from engine.contracts import design_hash


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Conflict(ValueError):
    pass


class Repository:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
                INSERT OR IGNORE INTO schema_version VALUES(1);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, created REAL NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, state TEXT NOT NULL,
                    owner TEXT, idempotency_key TEXT UNIQUE, request TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(job_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                    body TEXT NOT NULL, PRIMARY KEY(job_id,sequence));
                CREATE TABLE IF NOT EXISTS revisions(run_id TEXT NOT NULL, target_index INTEGER NOT NULL,
                    hash TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(run_id,target_index,hash));
                CREATE TABLE IF NOT EXISTS sessions(run_id TEXT NOT NULL, target_index INTEGER NOT NULL,
                    active_hash TEXT NOT NULL, PRIMARY KEY(run_id,target_index));
                CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    target_index INTEGER NOT NULL, revision_hash TEXT NOT NULL, body TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "workspace.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=15000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _owns(db,rid,owner):
        return owner is None or db.execute("SELECT 1 FROM jobs WHERE id=? AND owner=? AND state='running'",(rid,owner)).fetchone() is not None

    def owns(self,rid,owner):
        with self.connect() as db:return self._owns(db,rid,owner)

    def put_run(self, record, owner=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not self._owns(db,record['id'],owner):return False
            db.execute("INSERT INTO runs VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                       (record["id"], record["created"], encode(record)))
        return True

    def run(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT body FROM runs WHERE id=?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None

    def runs(self):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM runs ORDER BY created DESC LIMIT 100").fetchall()
        return [json.loads(row[0]) for row in rows]

    def create_job(self, rid, request, key=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                previous = db.execute("SELECT id,request FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
                if previous:
                    if previous[1] != encode(request):
                        raise Conflict("Idempotency key already identifies a different request")
                    return previous[0], False
            active = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
            if active >= 8:
                raise Conflict("Local job queue is full")
            body = {"id": rid, "run_id": rid, "state": "queued", "progress": 0,
                    "cancel": False, "result": None, "created": time.time(), "stage": "queued"}
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)", (rid, "queued", None, key, encode(request), encode(body)))
        return rid, True

    def job(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None

    def job_request(self, rid):
        with self.connect() as db:
            row = db.execute("SELECT request FROM jobs WHERE id=?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None

    def claim(self, rid, owner):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=? AND state='queued'", (rid,)).fetchone()
            if not row:
                return False
            body = {**json.loads(row[0]), "state": "running", "stage": "initializing", "owner": owner}
            db.execute("UPDATE jobs SET state='running',owner=?,body=? WHERE id=? AND state='queued'",
                       (owner, encode(body), rid))
            return True

    def update_job(self, rid, changes, owner=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state,owner,body FROM jobs WHERE id=?", (rid,)).fetchone()
            if not row or (owner is not None and row[1] != owner):
                return False
            if row[0] in ("done", "failed", "cancelled", "interrupted"):
                return False
            body = {**json.loads(row[2]), **changes}
            # A late completion must not turn a requested cancellation into done.
            if json.loads(row[2]).get("cancel") and changes.get("state") == "done":
                body["state"] = "cancelled"
            db.execute("UPDATE jobs SET state=?,body=? WHERE id=?", (body["state"], encode(body), rid))
        return True

    def interrupt_orphaned(self, live_ids=()):
        """Single local service process recovery; completed work stays attached."""
        recovered = []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id,body FROM jobs WHERE state IN ('queued','running')").fetchall()
            for row in rows:
                if row[0] in live_ids:
                    continue
                body = {**json.loads(row[1]), "state": "interrupted", "stage": "interrupted",
                        "error": "Local worker stopped. Partial results and revisions were preserved."}
                db.execute("UPDATE jobs SET state='interrupted',body=? WHERE id=?", (encode(body), row[0]))
                recovered.append(row[0])
        return recovered

    def requeue(self, rid):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT state,body FROM jobs WHERE id=?",(rid,)).fetchone()
            if not row or row[0] not in ('interrupted','cancelled','failed'):
                return False
            body={**json.loads(row[1]),'state':'queued','stage':'queued','cancel':False,'error':None,'owner':None}
            db.execute("UPDATE jobs SET state='queued',owner=NULL,body=? WHERE id=?",(encode(body),rid))
        return True

    def append_event(self, rid, event, owner=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._owns(db,rid,owner):return None
            seq = db.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM events WHERE job_id=?", (rid,)).fetchone()[0]
            body = {**event, "event_id": seq, "created": time.time(),"attempt_id":owner}
            db.execute("INSERT INTO events VALUES(?,?,?)", (rid, seq, encode(body)))
        return body

    def events(self, rid, after=-1):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 1000", (rid, after)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put_revision(self, rid, target_index, revision, activate=False, owner=None):
        hashed = design_hash(revision["design"])
        body = {**revision, "revision_hash": hashed, "run_id": rid, "target_index": target_index}
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not self._owns(db,rid,owner):return None
            db.execute("INSERT INTO revisions VALUES(?,?,?,?) ON CONFLICT(run_id,target_index,hash) DO UPDATE SET body=excluded.body",
                       (rid, target_index, hashed, encode(body)))
            if activate:
                db.execute("INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(run_id,target_index) DO UPDATE SET active_hash=excluded.active_hash",
                           (rid, target_index, hashed))
        return body

    def revision(self, rid, target_index, hashed):
        with self.connect() as db:
            row = db.execute("SELECT body FROM revisions WHERE run_id=? AND target_index=? AND hash=?", (rid, target_index, hashed)).fetchone()
        return json.loads(row[0]) if row else None

    def revisions(self, rid, target_index):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM revisions WHERE run_id=? AND target_index=? ORDER BY rowid", (rid, target_index)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def active_hash(self, rid, target_index):
        with self.connect() as db:
            row = db.execute("SELECT active_hash FROM sessions WHERE run_id=? AND target_index=?", (rid, target_index)).fetchone()
        return row[0] if row else None

    def compare_and_activate(self, rid, target_index, expected, hashed):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job=db.execute("SELECT state FROM jobs WHERE id=?",(rid,)).fetchone()
            if job and job[0] in ('queued','running'):raise Conflict('Stop generation before editing the active design')
            updated = db.execute("UPDATE sessions SET active_hash=? WHERE run_id=? AND target_index=? AND active_hash=?",
                                 (hashed, rid, target_index, expected)).rowcount
            if updated != 1:
                raise Conflict("The active design changed. Reload before applying this edit.")

    def feedback(self, rid, target_index, hashed, verdict, *, provenance="human", reason=None,
                 other_hash=None, key=None, reviewer_id="local", context=None):
        stable_key = key or hashlib.sha256(encode([rid, target_index, hashed, verdict, provenance, other_hash, reason,reviewer_id]).encode()).hexdigest()
        identity = {"run_id": rid, "target_index": target_index, "revision_hash": hashed,
                    "verdict": verdict, "provenance": provenance, "reason": reason, "other_hash": other_hash,
                    "reviewer_id":reviewer_id,"context":context or {}}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT body FROM feedback WHERE id=?", (stable_key,)).fetchone()
            if old:
                prior = json.loads(old[0])
                if any(prior.get(k) != v for k, v in identity.items()):
                    raise Conflict("Feedback idempotency key identifies a different judgment")
                return prior
            body = {**identity, "id": stable_key, "created": time.time()}
            db.execute("INSERT INTO feedback VALUES(?,?,?,?,?)", (stable_key, rid, target_index, hashed, encode(body)))
        return body

    def feedback_rows(self, rid):
        with self.connect() as db:
            rows = db.execute("SELECT body FROM feedback WHERE run_id=? ORDER BY rowid", (rid,)).fetchall()
        return [json.loads(row[0]) for row in rows]
