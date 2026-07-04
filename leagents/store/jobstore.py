"""SQLite-backed job store.

Loop state lives here — not in memory, not in framework checkpoints —
so runs are inspectable (`leagents status`) and resumable (DESIGN.md §4).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    stop_reason TEXT,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cycles (
    run_id TEXT NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    stage TEXT NOT NULL,
    decision TEXT,
    started_at REAL NOT NULL,
    finished_at REAL,
    PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    cycle_idx INTEGER NOT NULL,
    policy_type TEXT NOT NULL,
    path TEXT NOT NULL,
    blessed INTEGER NOT NULL DEFAULT 0,
    eval_json TEXT
);
"""


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # -- runs ------------------------------------------------------------
    def create_run(self, run_id: str, config_json: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, created_at, config_json) VALUES (?, ?, ?)",
                (run_id, time.time(), config_json),
            )

    def finish_run(self, run_id: str, status: str, stop_reason: str | None = None) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET status = ?, stop_reason = ? WHERE id = ?",
                (status, stop_reason, run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    # -- cycles ----------------------------------------------------------
    def start_cycle(self, run_id: str, idx: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO cycles (run_id, idx, stage, started_at) VALUES (?, ?, 'collect', ?)",
                (run_id, idx, time.time()),
            )

    def set_stage(self, run_id: str, idx: int, stage: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE cycles SET stage = ? WHERE run_id = ? AND idx = ?",
                (stage, run_id, idx),
            )

    def finish_cycle(self, run_id: str, idx: int, decision: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE cycles SET decision = ?, stage = 'done', finished_at = ? "
                "WHERE run_id = ? AND idx = ?",
                (decision, time.time(), run_id, idx),
            )

    def cycles_for(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM cycles WHERE run_id = ? ORDER BY idx", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- checkpoints -------------------------------------------------------
    def add_checkpoint(self, run_id: str, cycle_idx: int, policy_type: str, path: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO checkpoints (run_id, cycle_idx, policy_type, path) "
                "VALUES (?, ?, ?, ?)",
                (run_id, cycle_idx, policy_type, path),
            )
        return int(cur.lastrowid)

    def attach_eval(self, checkpoint_id: int, eval_report: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE checkpoints SET eval_json = ? WHERE id = ?",
                (json.dumps(eval_report), checkpoint_id),
            )

    def bless(self, run_id: str, checkpoint_id: int) -> None:
        """Mark one checkpoint as the run's blessed baseline (exclusive)."""
        with self._conn:
            self._conn.execute("UPDATE checkpoints SET blessed = 0 WHERE run_id = ?", (run_id,))
            self._conn.execute("UPDATE checkpoints SET blessed = 1 WHERE id = ?", (checkpoint_id,))

    def checkpoints_for(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle_idx", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def blessed_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ? AND blessed = 1", (run_id,)
        ).fetchone()
        return dict(row) if row else None
