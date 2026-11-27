import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def params():
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def small_train():
    """A small made-up training frame (the real CSVs are not in git)."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "pm25_today": rng.uniform(20, 300, n).round(1),
        "wind_kmph": rng.uniform(1, 30, n).round(1),
        "humidity_pct": rng.integers(20, 95, n),
        "temp_c": rng.uniform(5, 40, n).round(1),
        "traffic_index": rng.integers(10, 100, n),
        "stubble_fires": rng.integers(0, 400, n),
        "day_type": rng.choice(["weekday", "weekend", "festival"], n),
        "station_zone": rng.choice(["Central", "Residential", "Industrial"], n),
    })
    df["pm25_next_day"] = (300 - 3 * df.temp_c + 0.1 * df.stubble_fires
                           + rng.normal(0, 5, n)).round(1)
    return df
