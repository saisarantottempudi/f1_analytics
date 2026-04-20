"""Bulk-pull Ergast/Jolpica season data → Parquet under data/raw/ergast/.

Usage:
    python scripts/ingest_ergast.py --from 2000 --to 2025
    python scripts/ingest_ergast.py --from 2024 --to 2024 --skip-pitstops

Output files (one per dataset, all seasons concatenated):
    data/raw/ergast/schedule.parquet
    data/raw/ergast/results.parquet
    data/raw/ergast/qualifying.parquet   (2003+)
    data/raw/ergast/pitstops.parquet     (2012+)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data import ergast  # noqa: E402

OUT_DIR = ROOT / "data" / "raw" / "ergast"


def _write(rows: list[dict], path: Path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2000)
    ap.add_argument("--to", dest="end", type=int, default=2025)
    ap.add_argument("--skip-pitstops", action="store_true",
                    help="Pitstops are slow — one HTTP call per race.")
    args = ap.parse_args()

    all_schedule: list[dict] = []
    all_results: list[dict] = []
    all_quali: list[dict] = []
    all_pitstops: list[dict] = []

    seasons = range(args.start, args.end + 1)
    for year in tqdm(list(seasons), desc="Ergast seasons"):
        all_schedule.extend(ergast.season_schedule(year))
        all_results.extend(ergast.season_results(year))
        if year >= 2003:
            all_quali.extend(ergast.season_qualifying(year))
        if year >= 2012 and not args.skip_pitstops:
            all_pitstops.extend(ergast.season_pitstops(year))

    counts = {
        "schedule": _write(all_schedule, OUT_DIR / "schedule.parquet"),
        "results": _write(all_results, OUT_DIR / "results.parquet"),
        "qualifying": _write(all_quali, OUT_DIR / "qualifying.parquet"),
        "pitstops": _write(all_pitstops, OUT_DIR / "pitstops.parquet"),
    }
    print("\nWrote rows:")
    for name, n in counts.items():
        print(f"  {name:11s} {n:>7,d}")


if __name__ == "__main__":
    main()
