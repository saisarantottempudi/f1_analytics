"""Process-wide caches — features DataFrame and schedule, loaded once at startup."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
SCHEDULE_PATH = ROOT / "data" / "raw" / "ergast" / "schedule.parquet"


@lru_cache(maxsize=1)
def features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found — run `python scripts/build_features.py` first."
        )
    return pd.read_parquet(FEATURES_PATH)


@lru_cache(maxsize=1)
def schedule() -> pd.DataFrame:
    if not SCHEDULE_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(SCHEDULE_PATH)


def latest_driver_row(driver_id: str) -> pd.Series | None:
    """Most recent feature row per driver — used as the inference feature vector."""
    df = features()
    sub = df[df["driver_id"] == driver_id]
    if sub.empty:
        return None
    return sub.sort_values(["season", "round"]).iloc[-1]
