import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import events_sqlite


@pytest.fixture
def sqlite_events(tmp_path, monkeypatch):
    db_path = tmp_path / "events_test.db"
    monkeypatch.setattr(events_sqlite, "DB_PATH", str(db_path))
    return events_sqlite


def test_log_and_read_quality_event(sqlite_events):
    report = {"n_rows": 42, "passed": True, "drift_vs_previous_run": {}}
    event_id = sqlite_events.log_quality_event(report)
    assert event_id

    recent = sqlite_events.recent_quality_events(limit=5)
    assert len(recent) == 1
    assert recent[0]["n_rows"] == 42
    assert recent[0]["passed"] is True
    assert "logged_at" in recent[0]


def test_recent_quality_events_orders_newest_first(sqlite_events):
    sqlite_events.log_quality_event({"n_rows": 1, "passed": True})
    sqlite_events.log_quality_event({"n_rows": 2, "passed": False})
    recent = sqlite_events.recent_quality_events(limit=5)
    assert [r["n_rows"] for r in recent] == [2, 1]


def test_log_and_read_training_event(sqlite_events):
    event_id = sqlite_events.log_training_event(
        "run-uuid-1", "mlflow-run-1", {"metrics": {"accuracy": 0.83}},
    )
    assert event_id

    recent = sqlite_events.recent_training_events(limit=5)
    assert len(recent) == 1
    assert recent[0]["run_id"] == "run-uuid-1"
    assert recent[0]["mlflow_run_id"] == "mlflow-run-1"
    assert recent[0]["metrics"]["accuracy"] == 0.83


def test_recent_training_events_empty_when_none_logged(sqlite_events):
    assert sqlite_events.recent_training_events() == []


def test_get_db_ping_does_not_raise(sqlite_events):
    sqlite_events.get_db().command("ping")


def test_backend_selector_defaults_to_mongo(monkeypatch):
    monkeypatch.delenv("PERSISTENCE_BACKEND", raising=False)
    import src.backend as backend
    importlib.reload(backend)
    assert backend.log_quality_event.__module__ == "src.events"


def test_backend_selector_switches_to_sqlite(monkeypatch):
    monkeypatch.setenv("PERSISTENCE_BACKEND", "sqlite")
    import src.backend as backend
    importlib.reload(backend)
    assert backend.log_quality_event.__module__ == "src.events_sqlite"
    monkeypatch.delenv("PERSISTENCE_BACKEND", raising=False)
    importlib.reload(backend)
