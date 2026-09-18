import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quality import (
    check_drift, check_missing_values, check_ranges, check_schema,
    feature_stats, run_quality_checks,
)


def make_valid_df(n=50):
    return pd.DataFrame({
        "temperature_2m": [15.0] * n,
        "windspeed_10m": [10.0] * n,
        "windgusts_10m": [20.0] * n,
        "precipitation": [0.0] * n,
        "cloudcover": [50] * n,
        "hour": [12] * n,
        "month": [6] * n,
        "dayofweek": [2] * n,
        "airport_code": [0] * n,
    })


def test_check_schema_detects_missing_columns():
    df = make_valid_df().drop(columns=["cloudcover"])
    assert check_schema(df) == ["cloudcover"]


def test_check_schema_ok_on_valid_df():
    assert check_schema(make_valid_df()) == []


def test_check_missing_values_reports_only_nonzero():
    df = make_valid_df()
    df.loc[0:4, "temperature_2m"] = None
    report = check_missing_values(df)
    assert "temperature_2m" in report
    assert report["temperature_2m"] > 0
    assert "windspeed_10m" not in report


def test_check_ranges_flags_out_of_bound_cloudcover():
    df = make_valid_df()
    df.loc[0, "cloudcover"] = 150  # invalid: > 100
    violations = check_ranges(df)
    assert "cloudcover" in violations
    assert violations["cloudcover"]["n_violations"] == 1


def test_check_ranges_no_violation_on_valid_df():
    assert check_ranges(make_valid_df()) == {}


def test_check_drift_no_reference_returns_empty():
    stats = feature_stats(make_valid_df())
    assert check_drift(stats, None) == {}


def test_check_drift_detects_large_mean_shift():
    reference = {"temperature_2m": {"mean": 15.0, "std": 1.0}}
    current = {"temperature_2m": {"mean": 30.0, "std": 1.0}}  # +100%
    drift = check_drift(current, reference)
    assert "temperature_2m" in drift
    assert drift["temperature_2m"]["pct_change"] == pytest.approx(100.0)


def test_check_drift_ignores_small_shift():
    reference = {"temperature_2m": {"mean": 15.0, "std": 1.0}}
    current = {"temperature_2m": {"mean": 15.5, "std": 1.0}}  # ~3%
    assert check_drift(current, reference) == {}


def test_run_quality_checks_passes_on_clean_data():
    report = run_quality_checks(make_valid_df())
    assert report["passed"] is True
    assert report["n_rows"] == 50
    assert report["schema_missing_columns"] == []


def test_run_quality_checks_fails_on_range_violation():
    df = make_valid_df()
    df.loc[0, "cloudcover"] = 200
    report = run_quality_checks(df)
    assert report["passed"] is False
    assert "cloudcover" in report["range_violations"]


def test_run_quality_checks_fails_on_drift():
    df = make_valid_df()
    reference_stats = {"temperature_2m": {"mean": 100.0, "std": 1.0}}  # far from 15.0
    report = run_quality_checks(df, reference_stats)
    assert report["passed"] is False
    assert "temperature_2m" in report["drift_vs_previous_run"]
