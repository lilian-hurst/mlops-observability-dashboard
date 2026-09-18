"""Streamlit observability dashboard for the MLOps chain.

Read-only by design: it visualizes what the training pipeline (train.py) has
already logged to MLflow (structured run/metric history, Postgres-backed)
and MongoDB (data-quality reports and traceability events, schema-less).
This mirrors the brief of the Thales internship this project targets:
"développer et intégrer un outil de visualisation de données au cœur d'une
chaîne MLOps existante" -- quality, traceability and robustness of data and
models are the point, not just accuracy numbers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.backend import get_db, recent_quality_events, recent_training_events
from src.train import EXPERIMENT_NAME

st.set_page_config(page_title="MLOps Observability", page_icon="🛰️", layout="wide")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@st.cache_data(ttl=15)
def load_runs() -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=100,
    )
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id,
            "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
            "status": r.info.status,
            **{f"metric.{k}": v for k, v in r.data.metrics.items()},
            **{f"param.{k}": v for k, v in r.data.params.items()},
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=15)
def load_mongo_events():
    quality = recent_quality_events(limit=30)
    training = recent_training_events(limit=30)
    return quality, training


def connection_status():
    col1, col2 = st.columns(2)
    with col1:
        try:
            mlflow.tracking.MlflowClient().search_experiments()
            st.success("MLflow (Postgres) : connecté", icon="✅")
        except Exception as e:
            st.error(f"MLflow injoignable : {e}", icon="🚫")
    with col2:
        try:
            get_db().command("ping")
            st.success("MongoDB : connecté", icon="✅")
        except Exception as e:
            st.error(f"MongoDB injoignable : {e}", icon="🚫")


st.title("🛰️ MLOps Observability Dashboard")
st.caption(
    "Visualisation en direct d'une chaîne MLOps : historique des runs (MLflow / Postgres), "
    "qualité et dérive des données, traçabilité data → run (MongoDB)."
)

connection_status()
st.divider()

runs = load_runs()
quality_events, training_events = load_mongo_events()

if runs.empty:
    st.warning(
        "Aucun run trouvé pour l'expérience "
        f"'{EXPERIMENT_NAME}'. Lance `python3 src/train.py` pour en produire un."
    )
else:
    st.subheader("Historique des runs")
    metric_cols = [c for c in runs.columns if c.startswith("metric.")]
    display_cols = ["start_time", "run_id", "status"] + metric_cols
    st.dataframe(runs[display_cols].sort_values("start_time", ascending=False), width="stretch")

    st.subheader("Évolution des métriques")
    chart_metrics = [m for m in ["metric.accuracy", "metric.f1", "metric.roc_auc"] if m in runs.columns]
    if chart_metrics:
        chart_df = runs.sort_values("start_time")[["start_time"] + chart_metrics].set_index("start_time")
        st.line_chart(chart_df)

    latest = runs.sort_values("start_time", ascending=False).iloc[0]
    st.subheader(f"Dernier run — `{latest['run_id'][:12]}...`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{latest.get('metric.accuracy', 0):.2%}")
    c2.metric("F1", f"{latest.get('metric.f1', 0):.2f}")
    c3.metric("ROC-AUC", f"{latest.get('metric.roc_auc', 0):.2f}")
    c4.metric(
        "Qualité des données",
        "✅ OK" if latest.get("metric.data_quality_passed") == 1 else "⚠️ Alerte",
    )

st.divider()
st.subheader("Qualité et dérive des données (MongoDB)")
if quality_events:
    for ev in quality_events[:5]:
        status = "✅" if ev.get("passed") else "⚠️"
        with st.expander(f"{status} {ev.get('logged_at')} — {ev.get('n_rows')} lignes"):
            if ev.get("schema_missing_columns"):
                st.error(f"Colonnes manquantes : {ev['schema_missing_columns']}")
            if ev.get("range_violations"):
                st.warning(f"Valeurs hors plage : {ev['range_violations']}")
            if ev.get("drift_vs_previous_run"):
                st.warning(f"Dérive détectée : {ev['drift_vs_previous_run']}")
            if ev.get("passed"):
                st.success("Toutes les vérifications sont passées.")
            st.json(ev.get("missing_value_rates_pct", {}), expanded=False)
else:
    st.info("Aucun événement qualité pour l'instant.")

st.subheader("Traçabilité donnée → run (MongoDB)")
if training_events:
    trace_df = pd.DataFrame([
        {
            "logged_at": e.get("logged_at"),
            "mlflow_run_id": e.get("mlflow_run_id", "")[:12] + "...",
            "quality_event_id": e.get("quality_event_id", "")[:10] + "...",
            "date_range": e.get("date_range"),
            "accuracy": e.get("metrics", {}).get("accuracy"),
        }
        for e in training_events
    ])
    st.dataframe(trace_df, width="stretch")
else:
    st.info("Aucun événement de traçabilité pour l'instant.")
