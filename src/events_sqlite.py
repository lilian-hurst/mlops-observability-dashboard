"""SQLite-backed drop-in replacement for events.py, same function signatures.

Used only by the simplified single-container deployment (see README "Version
simplifiée en ligne"), which trades the real Postgres+MongoDB polyglot setup
for something that needs zero external services -- easier to deploy for free
on a single-process host like Render, at the cost of persistence (the SQLite
file lives on the container's ephemeral disk and resets on redeploy/restart,
so the demo container re-seeds itself with a fresh training run at startup).

The real MongoDB-backed version (events.py) is what docker-compose and the
Kubernetes manifests use, and is what was actually load-tested with two live
training runs -- see the main README for that verification. This module
exists purely to make the same dashboard.py / train.py code run against a
cheaper backend for the public demo, without duplicating their logic.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = os.environ.get("SQLITE_EVENTS_PATH", "data/events.db")


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS quality_events "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, logged_at TEXT, payload TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_events "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, logged_at TEXT, payload TEXT)"
    )
    return conn


def log_quality_event(report: dict[str, Any]) -> str:
    logged_at = datetime.now(timezone.utc).isoformat()
    doc = {**report, "logged_at": logged_at}
    conn = _connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO quality_events (logged_at, payload) VALUES (?, ?)",
            (logged_at, json.dumps(doc)),
        )
    conn.close()
    return str(cur.lastrowid)


def recent_quality_events(limit: int = 20) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, payload FROM quality_events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for row_id, payload in rows:
        doc = json.loads(payload)
        doc["_id"] = str(row_id)
        out.append(doc)
    return out


def log_training_event(run_id: str, mlflow_run_id: str, metadata: dict[str, Any]) -> str:
    logged_at = datetime.now(timezone.utc).isoformat()
    doc = {"run_id": run_id, "mlflow_run_id": mlflow_run_id, "logged_at": logged_at, **metadata}
    conn = _connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO training_events (logged_at, payload) VALUES (?, ?)",
            (logged_at, json.dumps(doc)),
        )
    conn.close()
    return str(cur.lastrowid)


def recent_training_events(limit: int = 20) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, payload FROM training_events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for row_id, payload in rows:
        doc = json.loads(payload)
        doc["_id"] = str(row_id)
        out.append(doc)
    return out


def get_db():
    """Health-check shim so dashboard.py's connection_status() works unchanged."""
    class _Ping:
        def command(self, *_args, **_kwargs):
            _connect().close()
            return {"ok": 1}
    return _Ping()
