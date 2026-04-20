"""Pull FastF1 per-lap telemetry → Parquet under data/raw/fastf1/.

FastF1 coverage begins ~2018. Earlier seasons return empty sessions.

Usage:
    python scripts/ingest_fastf1.py --from 2022 --to 2025
    python scripts/ingest_fastf1.py --from 2024 --to 2024 --sessions R  # race only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Silence FastF1's routine "no data" warnings for older / sprint-weekend edge cases.
warnings.filterwarnings("ignore", module="fastf1")

import fastf1  # noqa: E402
from fastf1.exceptions import DataNotLoadedError  # noqa: E402

CACHE_DIR = Path(os.getenv("FASTF1_CACHE_DIR", ROOT / "fastf1_cache"))
OUT_DIR = ROOT / "data" / "raw" / "fastf1"

LAP_COLS = [
    "Driver", "DriverNumber", "Team",
    "LapNumber", "LapTime", "Stint", "Compound", "TyreLife", "FreshTyre",
    "Position", "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Sector1Time", "Sector2Time", "Sector3Time",
]


def _load_session_safe(year: int, rnd: int, kind: str):
    try:
        s = fastf1.get_session(year, rnd, kind)
        s.load(laps=True, telemetry=False, weather=True, messages=False)
        return s
    except Exception:
        return None


def _laps_to_rows(session, year: int, rnd: int, kind: str) -> list[dict]:
    if session is None:
        return []
    try:
        laps = session.laps
    except DataNotLoadedError:
        return []
    if laps is None or laps.empty:
        return []
    df = laps.copy()
    keep = [c for c in LAP_COLS if c in df.columns]
    df = df[keep].copy()
    for c in ("LapTime", "PitInTime", "PitOutTime",
              "Sector1Time", "Sector2Time", "Sector3Time"):
        if c in df.columns:
            df[c] = df[c].dt.total_seconds()
    df.insert(0, "session", kind)
    df.insert(0, "round", rnd)
    df.insert(0, "season", year)
    return df.to_dict(orient="records")


def _weather_to_rows(session, year: int, rnd: int, kind: str) -> list[dict]:
    if session is None:
        return []
    try:
        wdf = session.weather_data
    except DataNotLoadedError:
        return []
    if wdf is None or wdf.empty:
        return []
    w = wdf.copy()
    w.insert(0, "session", kind)
    w.insert(0, "round", rnd)
    w.insert(0, "season", year)
    if "Time" in w.columns:
        w["Time"] = w["Time"].dt.total_seconds()
    return w.to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2022)
    ap.add_argument("--to", dest="end", type=int, default=2025)
    ap.add_argument("--sessions", default="Q,R",
                    help="Comma-separated: FP1,FP2,FP3,Q,S,R (default Q,R)")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]

    all_laps: list[dict] = []
    all_weather: list[dict] = []

    for year in range(args.start, args.end + 1):
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            print(f"[{year}] skipping — schedule load failed: {exc}")
            continue

        rounds = sched["RoundNumber"].tolist()
        for rnd in tqdm(rounds, desc=f"FastF1 {year}"):
            for kind in sessions:
                sess = _load_session_safe(year, rnd, kind)
                all_laps.extend(_laps_to_rows(sess, year, rnd, kind))
                all_weather.extend(_weather_to_rows(sess, year, rnd, kind))

    if all_laps:
        pd.DataFrame(all_laps).to_parquet(OUT_DIR / "laps.parquet", index=False)
    if all_weather:
        pd.DataFrame(all_weather).to_parquet(OUT_DIR / "weather.parquet", index=False)

    print(f"\nWrote {len(all_laps):,} lap rows · {len(all_weather):,} weather rows")


if __name__ == "__main__":
    main()
