"""Build the engineered feature table from raw Ergast Parquet.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --window 3    # shorter rolling window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.features.build import (  # noqa: E402
    build_feature_table, save_features,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=5,
                    help="Rolling-window size (in races) for form metrics.")
    args = ap.parse_args()

    print("Building feature table…")
    df = build_feature_table(rolling_window=args.window)
    out = save_features(df)
    print(f"\nWrote {len(df):,} rows × {df.shape[1]} cols → {out.relative_to(ROOT)}")
    print("\nSeason coverage:", df["season"].min(), "→", df["season"].max())
    print("Feature columns:")
    for c in df.columns:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
