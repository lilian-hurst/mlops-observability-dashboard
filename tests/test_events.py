import sys
from pathlib import Path

import mongomock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import events


@pytest.fixture
def fake_db(monkeypatch):
    client = mongomock.MongoClient()
    db = client[events.DB_NAME]
    monkeypatch.setattr(events, "get_db", lambda: db)
    return db


def test_log_and_read_quality_event(fake_db):
    report = {"n_rows": 42, "passed": True, "drift_vs_previous_run": {}}
    event_id = events.log_quality_event(report)
    assert event_id

    recent = events.recent_quality_events(limit=5)
    assert len(recent) == 1
    assert recent[0]["n_rows"] == 42
    assert recent[0]["passed"] is True
    assert "logged_at" in recent[0]


def test_recent_quality_events_orders_newest_first(fake_db):
    events.log_quality_event({"n_rows": 1, "passed": True})
    events.log_quality_event({"n_rows": 2, "passed": False})
    recent = events.recent_quality_events(limit=5)
    assert [r["n_rows"] for r in recent] == [2, 1]


def test_log_and_read_training_event(fake_db):
    event_id = events.log_training_event(
        "run-uuid-1", "mlflow-run-1", {"metrics": {"accuracy": 0.83}},
    )
    assert event_id

    recent = events.recent_training_events(limit=5)
    assert len(recent) == 1
    assert recent[0]["run_id"] == "run-uuid-1"
    assert recent[0]["mlflow_run_id"] == "mlflow-run-1"
    assert recent[0]["metrics"]["accuracy"] == 0.83


def test_recent_training_events_empty_when_none_logged(fake_db):
    assert events.recent_training_events() == []
