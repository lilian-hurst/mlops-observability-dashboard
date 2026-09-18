"""Persistence backend selector: MongoDB (real, tested with docker-compose)
or local SQLite (simplified, for the free single-container online demo).

Controlled by the PERSISTENCE_BACKEND env var: "mongo" (default) or "sqlite".
train.py and dashboard.py import log_quality_event/recent_quality_events/
log_training_event/recent_training_events/get_db from here instead of
choosing a module directly, so the exact same pipeline and dashboard code
runs unmodified against either backend.
"""
import os

if os.environ.get("PERSISTENCE_BACKEND", "mongo") == "sqlite":
    from src.events_sqlite import (  # noqa: F401
        get_db, log_quality_event, log_training_event,
        recent_quality_events, recent_training_events,
    )
else:
    from src.events import (  # noqa: F401
        get_db, log_quality_event, log_training_event,
        recent_quality_events, recent_training_events,
    )
