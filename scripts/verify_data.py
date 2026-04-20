"""Sanity-check the ingested Parquet files — row counts + date range per dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _report(path: Path) -> None:
    if not path.exists():
        print(f"  ❌ missing  {path.relative_to(ROOT)}")
        return
    df = pd.read_parquet(path)
    extras: list[str] = []
    if "season" in df.columns:
        extras.append(f"{df['season'].min()}–{df['season'].max()}")
    if "date" in df.columns:
        extras.append(f"{df['date'].min()} → {df['date'].max()}")
    suffix = f"  [{' · '.join(extras)}]" if extras else ""
    print(f"  ✅ {path.relative_to(ROOT)}: {len(df):,} rows{suffix}")


def main() -> None:
    print("Ergast:")
    for f in ("schedule", "results", "qualifying", "pitstops"):
        _report(RAW / "ergast" / f"{f}.parquet")
    print("\nFastF1:")
    for f in ("laps", "weather"):
        _report(RAW / "fastf1" / f"{f}.parquet")

    if not any((RAW / "ergast").glob("*.parquet")):
        print("\nNo data yet. Run:")
        print("  python scripts/ingest_ergast.py --from 2000 --to 2025")
        sys.exit(1)


if __name__ == "__main__":
    main()
