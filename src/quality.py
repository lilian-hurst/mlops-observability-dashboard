"""Data-quality and drift checks, run before every training job.

Pure functions (no DB access) so they're easy to unit test in isolation --
the actual persistence to MongoDB happens in events.py, kept separate on
purpose (traceability data, not core logic).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "temperature_2m", "windspeed_10m", "windgusts_10m", "precipitation",
    "cloudcover", "hour", "month", "dayofweek", "airport_code",
]

RANGE_CHECKS = {
    "cloudcover": (0, 100),
    "windspeed_10m": (0, None),
    "windgusts_10m": (0, None),
    "precipitation": (0, None),
    "hour": (0, 23),
    "month": (1, 12),
    "dayofweek": (0, 6),
}

# Relative change in mean beyond which a feature is flagged as "drifted"
# compared to a previous reference run.
DRIFT_THRESHOLD_PCT = 25.0


def check_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of missing expected columns (empty = OK)."""
    return [c for c in EXPECTED_COLUMNS if c not in df.columns]


def check_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """Return the fraction of missing values per column (only non-zero ones)."""
    rates = (df[EXPECTED_COLUMNS].isna().mean() * 100).round(2)
    return {col: float(rate) for col, rate in rates.items() if rate > 0}


def check_ranges(df: pd.DataFrame) -> dict[str, dict]:
    """Return, per column with out-of-range values, how many rows violate it."""
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        n_bad = int(mask.sum())
        if n_bad:
            violations[col] = {"n_violations": n_bad, "expected_range": [lo, hi]}
    return violations


def feature_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Mean/std snapshot per numeric feature -- used as the drift reference."""
    stats = {}
    for col in EXPECTED_COLUMNS:
        if col in df.columns:
            stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}
    return stats


def check_drift(current_stats: dict, reference_stats: dict | None) -> dict[str, dict]:
    """Compare current feature means to a reference run's means.

    Returns only the features whose mean moved by more than
    DRIFT_THRESHOLD_PCT relative to the reference. If there is no reference
    (first run ever), returns an empty dict -- nothing to compare against.
    """
    if not reference_stats:
        return {}
    drifted = {}
    for col, cur in current_stats.items():
        ref = reference_stats.get(col)
        if not ref or ref["mean"] == 0:
            continue
        pct_change = abs(cur["mean"] - ref["mean"]) / abs(ref["mean"]) * 100
        if pct_change > DRIFT_THRESHOLD_PCT:
            drifted[col] = {
                "reference_mean": ref["mean"], "current_mean": cur["mean"],
                "pct_change": round(pct_change, 1),
            }
    return drifted


def run_quality_checks(df: pd.DataFrame, reference_stats: dict | None = None) -> dict[str, Any]:
    """Run all checks and return one consolidated report, ready to log."""
    missing_cols = check_schema(df)
    missing_values = check_missing_values(df)
    range_violations = check_ranges(df)
    stats = feature_stats(df)
    drift = check_drift(stats, reference_stats)

    passed = not missing_cols and not range_violations and not drift
    return {
        "n_rows": len(df),
        "schema_missing_columns": missing_cols,
        "missing_value_rates_pct": missing_values,
        "range_violations": range_violations,
        "feature_stats": stats,
        "drift_vs_previous_run": drift,
        "passed": passed,
    }
