"""Storage — G1 permutation: SQLite WAL, embeddings as packed f32 BLOBs.

taste-6 adds the working_set table (H6) — the agent's bounded curated
context, written only by the owner's resolve path.

All writes go through the reentrant `tx()` context manager: an outer tx
holds the transaction open so multi-write operations (amendment apply)
commit atomically — a crash can never half-apply.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import (
    Category,
    DecisionRecord,
    LayerState,
    Polarity,
    Provenance,
    Standard,
    WorkingSetEntry,
    WorkingSetLayerInfo,
)


class HoneBudgetExhausted(Exception):
    """A task has used its hone calls. The cap is per task, across segments."""


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "taste6.db"
WS_MAX_SLOTS = 20  # H6: curation is forced by scarcity

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-64000;

CREATE TABLE IF NOT EXISTS standards (
  id          TEXT PRIMARY KEY,
  profile_id  TEXT NOT NULL,
  category    TEXT NOT NULL,
  polarity    TEXT NOT NULL,
  directive   TEXT NOT NULL,
  rationale   TEXT NOT NULL DEFAULT '',
  confidence  REAL NOT NULL DEFAULT 0.5,
  provenance  TEXT NOT NULL,
  state       TEXT NOT NULL DEFAULT 'provisional',
  supersedes  TEXT NOT NULL DEFAULT '[]',
  created     TEXT NOT NULL,
  updated     TEXT NOT NULL,
  variant_meta TEXT NOT NULL DEFAULT '{}',
  embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_standards_scope
  ON standards(profile_id, state, category, polarity);

CREATE TABLE IF NOT EXISTS decisions (
  id          TEXT PRIMARY KEY,
  profile_id  TEXT NOT NULL,
  task_ctx    TEXT NOT NULL DEFAULT '{}',
  candidates  TEXT NOT NULL DEFAULT '[]',
  pick        TEXT NOT NULL,
  rationale   TEXT NOT NULL,
  agent_role  TEXT NOT NULL DEFAULT '',
  provenance  TEXT NOT NULL DEFAULT 'owner_pick',
  ts          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inferred_queue (
  id          TEXT PRIMARY KEY,
  profile_id  TEXT NOT NULL,
  kind        TEXT NOT NULL,           -- mirror | answer | amendment
  payload     TEXT NOT NULL,           -- json
  ts          TEXT NOT NULL,
  resolved    INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS standards_fts USING fts5(
  id UNINDEXED, directive, rationale
);

CREATE TABLE IF NOT EXISTS working_set (
  profile_id  TEXT NOT NULL,
  standard_id TEXT NOT NULL,
  slot        INTEGER NOT NULL,
  weight      REAL NOT NULL,
  pinned_at   TEXT NOT NULL,
  PRIMARY KEY (profile_id, slot)
);

CREATE INDEX IF NOT EXISTS idx_ws_std
  ON working_set(profile_id, standard_id);

CREATE TABLE IF NOT EXISTS hone_budgets (
  profile_id    TEXT NOT NULL,
  task_key      TEXT NOT NULL,
  reservations  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (profile_id, task_key)
);
"""


def _pack(vec: list[float] | None) -> bytes | None:
    return None if vec is None else struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class Store:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or os.environ.get("TASTE6_DB", DEFAULT_DB))
        path.parent.mkdir(parents=True, exist_ok=True)
        # single-process service: one connection, writes serialized by lock,
        # cross-thread because the ASGI worker owns a different thread
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.RLock()  # reentrant: tx() nests inside tx()
        self._depth = 0
        self.db.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        """Atomic write scope — outermost tx commits, exception rolls back."""
        with self._lock:
            self._depth += 1
            try:
                yield
                if self._depth == 1:
                    self.db.commit()
            except BaseException:
                if self._depth == 1:
                    self.db.rollback()
                raise
            finally:
                self._depth -= 1

    def reserve_hone(self, profile_id: str, task_key: str, limit: int = 15) -> int:
        """Count one hone call. Raise once this task is past `limit`."""
        with self.tx():
            self.db.execute(
                "INSERT OR IGNORE INTO hone_budgets"
                " (profile_id, task_key, reservations) VALUES (?,?,0)",
                (profile_id, task_key))
            row = self.db.execute(
                "SELECT reservations FROM hone_budgets"
                " WHERE profile_id=? AND task_key=?",
                (profile_id, task_key)).fetchone()
            current = int(row["reservations"])
            if current >= limit:
                raise HoneBudgetExhausted("The task hone budget is exhausted.")
            reservation = current + 1
            self.db.execute(
                "UPDATE hone_budgets SET reservations=?"
                " WHERE profile_id=? AND task_key=?",
                (reservation, profile_id, task_key))
            return reservation

    # ------------------------------------------------------------ standards
    def put_standard(self, s: Standard, embedding: list[float] | None = None):
        with self.tx():
            self.db.execute(
                "INSERT OR REPLACE INTO standards"
                " (id, profile_id, category, polarity, directive, rationale,"
                "  confidence, provenance, state, supersedes, created, updated,"
                "  variant_meta, embedding)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.id, s.profile_id, s.category.value, s.polarity.value,
                 s.directive, s.rationale, s.confidence, s.provenance.value,
                 s.state.value, json.dumps(s.supersedes),
                 s.created.isoformat(), s.updated.isoformat(),
                 json.dumps(s.variant_meta), _pack(embedding)),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO standards_fts (id, directive, rationale)"
                " VALUES (?,?,?)", (s.id, s.directive, s.rationale))

    def get_standard(self, sid: str) -> Standard | None:
        row = self.db.execute(
            "SELECT * FROM standards WHERE id=?", (sid,)).fetchone()
        return self._row_to_standard(row) if row else None

    def list_standards(self, profile_id: str = "owner",
                       state: LayerState | None = None,
                       category: Category | None = None,
                       polarity: Polarity | None = None,
                       with_vectors: bool = False
                       ) -> list[Standard] | list[tuple[Standard, list[float] | None]]:
        q = "SELECT * FROM standards WHERE profile_id=?"
        args: list = [profile_id]
        if state:
            q += " AND state=?"
            args.append(state.value)
        if category:
            q += " AND category=?"
            args.append(category.value)
        if polarity:
            q += " AND polarity=?"
            args.append(polarity.value)
        rows = self.db.execute(q, args).fetchall()
        if with_vectors:
            return [(self._row_to_standard(r), _unpack(r["embedding"]))
                    for r in rows]
        return [self._row_to_standard(r) for r in rows]

    def set_state(self, sid: str, state: LayerState):
        with self.tx():
            self.db.execute(
                "UPDATE standards SET state=?, updated=datetime('now') WHERE id=?",
                (state.value, sid))

    def set_variant_meta(self, sid: str, meta: dict):
        with self.tx():
            self.db.execute(
                "UPDATE standards SET variant_meta=?, updated=datetime('now')"
                " WHERE id=?",
                (json.dumps(meta), sid))

    def set_rationale(self, sid: str, rationale: str):
        with self.tx():
            self.db.execute(
                "UPDATE standards SET rationale=?, updated=datetime('now')"
                " WHERE id=?", (rationale, sid))
            self.db.execute(
                "UPDATE standards_fts SET rationale=? WHERE id=?",
                (rationale, sid))

    def set_confidence(self, sid: str, conf: float):
        with self.tx():
            self.db.execute(
                "UPDATE standards SET confidence=?, updated=datetime('now')"
                " WHERE id=?", (conf, sid))

    @staticmethod
    def _row_to_standard(r: sqlite3.Row) -> Standard:
        return Standard(
            id=r["id"], profile_id=r["profile_id"],
            category=Category(r["category"]), polarity=Polarity(r["polarity"]),
            directive=r["directive"], rationale=r["rationale"],
            confidence=r["confidence"], provenance=Provenance(r["provenance"]),
            state=LayerState(r["state"]),
            supersedes=json.loads(r["supersedes"]),
            created=r["created"], updated=r["updated"],
            variant_meta=json.loads(r["variant_meta"]),
        )

    # ------------------------------------------------------------ decisions
    def put_decision(self, d: DecisionRecord):
        with self.tx():
            self.db.execute(
                "INSERT INTO decisions"
                " (id, profile_id, task_ctx, candidates, pick, rationale,"
                "  agent_role, provenance, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (d.id, d.profile_id, json.dumps(d.task_ctx),
                 json.dumps([c.model_dump() for c in d.candidates]),
                 d.pick, d.rationale, d.agent_role, d.provenance,
                 d.ts.isoformat()),
            )

    def list_decisions(self, profile_id: str = "owner") -> list[DecisionRecord]:
        rows = self.db.execute(
            "SELECT * FROM decisions WHERE profile_id=? ORDER BY ts",
            (profile_id,)).fetchall()
        return [DecisionRecord(
            id=r["id"], profile_id=r["profile_id"],
            task_ctx=json.loads(r["task_ctx"]),
            candidates=[],  # hydrated on demand — list view stays light
            pick=r["pick"], rationale=r["rationale"],
            agent_role=r["agent_role"], provenance=r["provenance"],
            ts=r["ts"]) for r in rows]

    # ------------------------------------------------------------ inferred queue
    def enqueue_inferred(self, iid: str, profile_id: str, kind: str,
                         payload: dict):
        with self.tx():
            self.db.execute(
                "INSERT INTO inferred_queue (id, profile_id, kind, payload, ts)"
                " VALUES (?,?,?,?,?)",
                (iid, profile_id, kind, json.dumps(payload),
                 datetime.now(UTC).isoformat()))

    def pending_inferred(self, profile_id: str = "owner") -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM inferred_queue WHERE profile_id=? AND resolved=0"
            " ORDER BY ts", (profile_id,)).fetchall()
        return [{"id": r["id"], "kind": r["kind"],
                 "payload": json.loads(r["payload"]), "ts": r["ts"]}
                for r in rows]

    def get_inferred(self, iid: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM inferred_queue WHERE id=?", (iid,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "profile_id": row["profile_id"],
                "kind": row["kind"], "payload": json.loads(row["payload"]),
                "ts": row["ts"], "resolved": bool(row["resolved"])}

    def resolve_inferred(self, iid: str):
        with self.tx():
            self.db.execute(
                "UPDATE inferred_queue SET resolved=1 WHERE id=?", (iid,))

    # ------------------------------------------------------------ working set (H6)
    def put_ws_entry(self, profile_id: str, standard_id: str,
                     weight: float = 1.0) -> WorkingSetEntry:
        """Idempotent per standard: re-pin updates weight in the same slot."""
        with self.tx():
            row = self.db.execute(
                "SELECT slot, pinned_at FROM working_set"
                " WHERE profile_id=? AND standard_id=?",
                (profile_id, standard_id)).fetchone()
            if row:
                entry = WorkingSetEntry(
                    profile_id=profile_id, standard_id=standard_id,
                    slot=row["slot"], weight=weight,
                    pinned_at=datetime.now(UTC))
                self.db.execute(
                    "UPDATE working_set SET weight=?, pinned_at=?"
                    " WHERE profile_id=? AND slot=?",
                    (weight, entry.pinned_at.isoformat(),
                     profile_id, row["slot"]))
                return entry
            used = {r[0] for r in self.db.execute(
                "SELECT slot FROM working_set WHERE profile_id=?",
                (profile_id,))}
            if len(used) >= WS_MAX_SLOTS:
                raise ValueError(
                    f"working set full ({WS_MAX_SLOTS} slots) — demote first")
            slot = min(set(range(1, WS_MAX_SLOTS + 1)) - used)
            entry = WorkingSetEntry(
                profile_id=profile_id, standard_id=standard_id,
                slot=slot, weight=weight, pinned_at=datetime.now(UTC))
            self.db.execute(
                "INSERT INTO working_set"
                " (profile_id, standard_id, slot, weight, pinned_at)"
                " VALUES (?,?,?,?,?)",
                (profile_id, standard_id, slot, weight,
                 entry.pinned_at.isoformat()))
            return entry

    def get_working_set(self, profile_id: str = "owner"
                        ) -> list[WorkingSetLayerInfo]:
        """WS rows joined to layer state — the self-healing read path.

        Rows whose layer is not active (or no longer exists) are never
        returned: the WS can hold no dead layers at any resting state,
        even if a writer somewhere missed a purge hook.
        """
        rows = self.db.execute(
            "SELECT w.slot, w.standard_id, w.weight, w.pinned_at,"
            " s.directive, s.category, s.polarity, s.state, s.confidence"
            " FROM working_set w"
            " JOIN standards s ON s.id = w.standard_id AND s.profile_id = w.profile_id"
            " WHERE w.profile_id=? AND s.state='active'"
            " ORDER BY w.weight DESC, w.slot ASC",
            (profile_id,)).fetchall()
        return [WorkingSetLayerInfo(
            slot=r["slot"], standard_id=r["standard_id"], weight=r["weight"],
            pinned_at=r["pinned_at"], directive=r["directive"],
            category=Category(r["category"]), polarity=Polarity(r["polarity"]),
            state=LayerState(r["state"]), confidence=r["confidence"],
        ) for r in rows]

    def del_ws_entry(self, profile_id: str, slot: int):
        with self.tx():
            self.db.execute(
                "DELETE FROM working_set WHERE profile_id=? AND slot=?",
                (profile_id, slot))

    def purge_ws_standard(self, profile_id: str, standard_id: str):
        """Key rule: supersede/retire evicts the layer's slot automatically."""
        with self.tx():
            self.db.execute(
                "DELETE FROM working_set WHERE profile_id=? AND standard_id=?",
                (profile_id, standard_id))

    def ws_slot_count(self, profile_id: str = "owner") -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM working_set WHERE profile_id=?",
            (profile_id,)).fetchone()[0]
