"""Real training data: historical weather from Open-Meteo (free, keyless API).

This reuses the same weather-risk methodology validated in the
`flight-disruption-risk` project (proxy label from documented aviation
weather thresholds, forecasting target at +3h to avoid data leakage -- see
that project's README for the full write-up). It is intentionally kept small
here: the point of *this* project is the MLOps tooling around a pipeline
(tracking, data-quality logging, observability dashboard), not the modeling
itself.
"""
from __future__ import annotations

import requests
import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,windspeed_10m,windgusts_10m,precipitation,cloudcover"

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
    params = {
        "latitude": lat, "longitude": lon, "start_date": start_date,
        "end_date": end_date, "hourly": HOURLY_VARS, "timezone": "UTC",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def build_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for icao, info in AIRPORTS.items():
        payload = fetch_historical_weather(info["lat"], info["lon"], start_date, end_date)
        df = pd.DataFrame(payload["hourly"])
        df["airport"] = icao
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data["time"])
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
