"""Real training data: historical weather from Open-Meteo (free, keyless API).

This reuses the same weather-risk methodology validated in the
`flight-disruption-risk` project (proxy label from documented aviation
weather thresholds, forecasting target at +3h to avoid data leakage -- see
that project's README for the full write-up). It is intentionally kept small
here: the point of *this* project is the MLOps tooling around a pipeline
(tracking, data-quality logging, observability dashboard), not the modeling
itself.

Resilience: Open-Meteo's free tier can return 429 (rate limited) when called
from a shared hosting IP -- observed in production on Render, where other
tenants on the same egress IP can exhaust the per-IP quota. fetch_historical_
weather() retries with backoff, and if it still fails, build_dataset() falls
back to a bundled real snapshot (data/fallback_weather.csv, fetched once and
committed to the repo) rather than leaving the dashboard empty. Every use of
the fallback is logged loudly so it's never silently mistaken for live data.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,windspeed_10m,windgusts_10m,precipitation,cloudcover"
FALLBACK_CSV = Path(__file__).resolve().parent.parent / "data" / "fallback_weather.csv"

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = (5, 15, 30)

AIRPORTS = {
    "LFPG": {"name": "Paris CDG", "lat": 49.0097, "lon": 2.5479},
    "LFMN": {"name": "Nice", "lat": 43.6584, "lon": 7.2159},
    "EDDF": {"name": "Frankfurt", "lat": 50.0379, "lon": 8.5622},
    "EGLL": {"name": "London Heathrow", "lat": 51.4700, "lon": -0.4543},
}

PRECIP_THRESHOLD_MM = 2.0
GUST_THRESHOLD_KMH = 50.0
CLOUDCOVER_THRESHOLD_PCT = 90.0
HORIZON_HOURS = 3

FEATURES = [
    "temperature_2m", "windspeed_10m", "windgusts_10m", "precipitation",
    "cloudcover", "hour", "month", "dayofweek", "airport_code",
]
TARGET = "disruption_risk_future"


def label_disruption_risk(row: pd.Series) -> int:
    return int(
        row["precipitation"] > PRECIP_THRESHOLD_MM
        or row["windgusts_10m"] > GUST_THRESHOLD_KMH
        or row["cloudcover"] > CLOUDCOVER_THRESHOLD_PCT
    )


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Fetch hourly historical weather, retrying transient errors (429/5xx)."""
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date,
        "end_date": end_date, "hourly": HOURLY_VARS, "timezone": "UTC",
    }
    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None
            if status not in (429, 500, 502, 503, 504) or attempt == RETRY_ATTEMPTS - 1:
                raise
            wait_s = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
            logger.warning(
                "Open-Meteo a répondu %s (tentative %d/%d), nouvel essai dans %ds...",
                status, attempt + 1, RETRY_ATTEMPTS, wait_s,
            )
            time.sleep(wait_s)
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            wait_s = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
            logger.warning(
                "Erreur réseau Open-Meteo (tentative %d/%d) : %s -- nouvel essai dans %ds...",
                attempt + 1, RETRY_ATTEMPTS, e, wait_s,
            )
            time.sleep(wait_s)
    raise last_error  # pragma: no cover -- unreachable, loop always returns or raises


def _load_fallback(icao: str) -> pd.DataFrame:
    """Real weather data fetched once and bundled with the repo, used only
    when the live Open-Meteo call fails after all retries. Never silent:
    callers must log that they're using it."""
    if not FALLBACK_CSV.exists():
        raise FileNotFoundError(f"Pas de secours disponible : {FALLBACK_CSV} est absent")
    df = pd.read_csv(FALLBACK_CSV, parse_dates=["time"])
    subset = df[df["airport"] == icao].drop(columns=["airport"])
    if subset.empty:
        raise ValueError(f"Le fichier de secours ne contient aucune donnée pour {icao}")
    return subset


def build_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    used_fallback_for = []
    for icao, info in AIRPORTS.items():
        try:
            payload = fetch_historical_weather(info["lat"], info["lon"], start_date, end_date)
            df = pd.DataFrame(payload["hourly"])
            df["time"] = pd.to_datetime(df["time"])
        except requests.exceptions.RequestException as e:
            logger.warning(
                "Échec définitif de la récupération météo en direct pour %s (%s) : %s -- "
                "utilisation des données de secours (data/fallback_weather.csv).",
                icao, info["name"], e,
            )
            df = _load_fallback(icao)
            used_fallback_for.append(icao)
        df["airport"] = icao
        frames.append(df)

    if used_fallback_for:
        print(
            f"ATTENTION : données de secours (pas en direct) utilisées pour : "
            f"{used_fallback_for} -- voir data/fallback_weather.csv"
        )

    data = pd.concat(frames, ignore_index=True)
    data["hour"] = data["time"].dt.hour
    data["month"] = data["time"].dt.month
    data["dayofweek"] = data["time"].dt.dayofweek
    data = data.sort_values(["airport", "time"]).reset_index(drop=True)
    data["disruption_risk_now"] = data.apply(label_disruption_risk, axis=1)
    data["disruption_risk_future"] = (
        data.groupby("airport")["disruption_risk_now"].shift(-HORIZON_HOURS)
    )
    data = data.dropna(subset=["disruption_risk_future"]).reset_index(drop=True)
    data["disruption_risk_future"] = data["disruption_risk_future"].astype(int)

    codes = {icao: i for i, icao in enumerate(sorted(AIRPORTS))}
    data["airport_code"] = data["airport"].map(codes)
    return data
