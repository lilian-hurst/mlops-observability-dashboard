import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    AIRPORTS, CLOUDCOVER_THRESHOLD_PCT, GUST_THRESHOLD_KMH,
    PRECIP_THRESHOLD_MM, label_disruption_risk,
)


def test_airports_reference_data_is_well_formed():
    assert len(AIRPORTS) >= 2
    for icao, info in AIRPORTS.items():
        assert len(icao) == 4
        assert -90 <= info["lat"] <= 90


@pytest.mark.parametrize(
    "precip, gust, cloud, expected",
    [
        (0.0, 0.0, 0, 0),
        (PRECIP_THRESHOLD_MM + 0.1, 0.0, 0, 1),
        (0.0, GUST_THRESHOLD_KMH + 0.1, 0, 1),
        (0.0, 0.0, CLOUDCOVER_THRESHOLD_PCT + 1, 1),
        (PRECIP_THRESHOLD_MM, GUST_THRESHOLD_KMH, CLOUDCOVER_THRESHOLD_PCT, 0),
    ],
)
def test_label_disruption_risk_thresholds(precip, gust, cloud, expected):
    row = pd.Series({"precipitation": precip, "windgusts_10m": gust, "cloudcover": cloud})
    assert label_disruption_risk(row) == expected
