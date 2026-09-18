"""MongoDB access layer for unstructured operational events.

This is the "NoSQL" half of the polyglot-persistence design (Postgres holds
MLflow's structured run/metric records; MongoDB holds free-form, schema-less
events: data-quality reports and traceability logs). Keeping these in a
document store rather than forcing them into relational tables matches how
this kind of data actually behaves in practice -- quality-check payloads
differ from one dataset version to the next, and you don't want a schema
migration every time you add a new check.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient, DESCENDING

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "mlops_observability"


def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return client[DB_NAME]


def log_quality_event(report: dict[str, Any]) -> str:
    """Store a data-quality report (see quality.py) and return its id."""
    db = get_db()
    report = {**report, "logged_at": datetime.now(timezone.utc)}
    result = db.quality_events.insert_one(report)
    return str(result.inserted_id)


def recent_quality_events(limit: int = 20) -> list[dict]:
    db = get_db()
    # Sort by _id (monotonically increasing, finer-grained than logged_at) as
    # a tiebreaker: two events logged within the same millisecond would
    # otherwise sort non-deterministically on "logged_at" alone.
    docs = list(
        db.quality_events.find()
        .sort([("logged_at", DESCENDING), ("_id", DESCENDING)])
        .limit(limit)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


def log_training_event(run_id: str, mlflow_run_id: str, metadata: dict[str, Any]) -> str:
    """Traceability record: which dataset/run produced which MLflow run."""
    db = get_db()
    doc = {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "logged_at": datetime.now(timezone.utc),
        **metadata,
    }
    result = db.training_events.insert_one(doc)
    return str(result.inserted_id)


def recent_training_events(limit: int = 20) -> list[dict]:
    db = get_db()
    docs = list(
        db.training_events.find()
        .sort([("logged_at", DESCENDING), ("_id", DESCENDING)])
        .limit(limit)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs
