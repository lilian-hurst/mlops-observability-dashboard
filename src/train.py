"""Training entrypoint: build data -> quality checks -> train -> track.

Every run does, in order:
1. Pull real historical weather data (data.py).
2. Run data-quality / drift checks against the most recent previous run's
   feature statistics (quality.py), log the report to MongoDB (events.py).
3. Train an XGBoost classifier and log params/metrics/model to MLflow
   (tracking server backed by Postgres -- see docker-compose.yml).
4. Log a traceability record linking the MongoDB quality report to the
   MLflow run id, so the dashboard can show "this model came from this data,
   which passed/failed these checks" end to end.

Usage:
    python3 src/train.py --start 2026-06-01 --end 2026-09-17
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.data import FEATURES, TARGET, build_dataset
from src.events import log_quality_event, log_training_event, recent_training_events
from src.quality import run_quality_checks

EXPERIMENT_NAME = "flight-weather-risk"


def previous_reference_stats() -> dict | None:
    """Fetch the feature stats logged by the most recent previous run, if any."""
    events = recent_training_events(limit=1)
    if not events:
        return None
    return events[0].get("feature_stats")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-09-17")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"-> Récupération des données réelles ({args.start} -> {args.end})...")
    df = build_dataset(args.start, args.end)

    print("-> Vérifications qualité / dérive des données...")
    reference_stats = previous_reference_stats()
    quality_report = run_quality_checks(df, reference_stats)
    quality_event_id = log_quality_event(quality_report)
    print(f"   Rapport qualité : passed={quality_report['passed']} "
          f"(id Mongo: {quality_event_id})")
    if quality_report["drift_vs_previous_run"]:
        print(f"   Dérive détectée sur : {list(quality_report['drift_vs_previous_run'])}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    run_uuid = str(uuid.uuid4())
    with mlflow.start_run(run_name=run_uuid) as run:
        params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "random_state": 42}
        mlflow.log_params(params)
        mlflow.log_param("date_range", f"{args.start} -> {args.end}")
        mlflow.log_param("n_rows_total", len(df))

        model = xgb.XGBClassifier(eval_metric="logloss", **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_metric("data_quality_passed", int(quality_report["passed"]))
        mlflow.xgboost.log_model(model, name="model")

        log_training_event(run_uuid, run.info.run_id, {
            "quality_event_id": quality_event_id,
            "feature_stats": quality_report["feature_stats"],
            "metrics": metrics,
            "date_range": f"{args.start} -> {args.end}",
        })

        print(f"-> Run MLflow : {run.info.run_id}")
        for k, v in metrics.items():
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
