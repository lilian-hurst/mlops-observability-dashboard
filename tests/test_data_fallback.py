import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data


def test_fallback_csv_exists_and_has_expected_columns():
    assert data.FALLBACK_CSV.exists(), "data/fallback_weather.csv doit être versionné"
    df = pd.read_csv(data.FALLBACK_CSV)
    for col in ["time", "temperature_2m", "windspeed_10m", "windgusts_10m",
                "precipitation", "cloudcover", "airport"]:
        assert col in df.columns
    for icao in data.AIRPORTS:
        assert icao in df["airport"].unique(), f"{icao} manquant dans le fichier de secours"


def test_load_fallback_returns_data_for_known_airport():
    df = data._load_fallback("LFPG")
    assert not df.empty
    assert "airport" not in df.columns  # dropped, re-added by the caller


def test_load_fallback_raises_for_unknown_airport():
    with pytest.raises(ValueError):
        data._load_fallback("ZZZZ")


def test_fetch_historical_weather_retries_on_429_then_succeeds():
    ok_response = requests.Response()
    ok_response.status_code = 200
    ok_response._content = b'{"hourly": {"time": []}}'

    error_response = requests.Response()
    error_response.status_code = 429

    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return error_response if call_count["n"] == 1 else ok_response

    with patch("src.data.requests.get", side_effect=fake_get), \
         patch("src.data.time.sleep") as mock_sleep:
        result = data.fetch_historical_weather(43.6, 7.2, "2026-06-01", "2026-06-02")

    assert result == {"hourly": {"time": []}}
    assert call_count["n"] == 2
    mock_sleep.assert_called_once()


def test_fetch_historical_weather_gives_up_after_max_retries():
    error_response = requests.Response()
    error_response.status_code = 429

    with patch("src.data.requests.get", return_value=error_response), \
         patch("src.data.time.sleep"):
        with pytest.raises(requests.exceptions.HTTPError):
            data.fetch_historical_weather(43.6, 7.2, "2026-06-01", "2026-06-02")


def test_build_dataset_falls_back_when_live_fetch_fails():
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("simulated failure")

    with patch("src.data.fetch_historical_weather", side_effect=boom):
        df = data.build_dataset("2026-07-01", "2026-09-17")

    assert not df.empty
    assert set(df["airport"].unique()) == set(data.AIRPORTS)
    assert data.TARGET in df.columns


def test_build_dataset_uses_live_data_when_available():
    fake_payload = {
        "hourly": {
            "time": [f"2026-06-01T{h:02d}:00" for h in range(10)],
            "temperature_2m": [15.0] * 10,
            "windspeed_10m": [10.0] * 10,
            "windgusts_10m": [20.0] * 10,
            "precipitation": [0.0] * 10,
            "cloudcover": [50] * 10,
        }
    }
    with patch("src.data.fetch_historical_weather", return_value=fake_payload):
        df = data.build_dataset("2026-06-01", "2026-06-01")

    # Live data used, not the fallback file -- rows come from the tiny fake payload
    assert len(df) == (10 - data.HORIZON_HOURS) * len(data.AIRPORTS)
