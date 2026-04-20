"""Demo: run a 1 000-race Monte Carlo sim and print win / podium / points probs.

Builds a grid from the latest row of features.parquet (one row per driver at
the final race of the dataset) and turns it into a `DriverState` list.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --sims 500 --laps 58
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.monte_carlo import (  # noqa: E402
    DriverState, RaceConfig, monte_carlo, summary,
)

FEATURES = ROOT / "data" / "processed" / "features.parquet"


def build_grid(df: pd.DataFrame) -> list[DriverState]:
    last_race_key = df[["season", "round"]].drop_duplicates().iloc[-1]
    sl = df[(df["season"] == last_race_key["season"])
            & (df["round"] == last_race_key["round"])].copy()
    sl = sl.sort_values("grid", na_position="last").head(20)

    # Map recent-form avg finish → pace gap vs leader (~0.2s per position).
    leader_form = sl["drv_rollN_avg_finish_5"].min()
    drivers: list[DriverState] = []
    for _, row in sl.iterrows():
        form = row.get("drv_rollN_avg_finish_5")
        pace = 0.0 if pd.isna(form) else 0.2 * (form - leader_form)
        dnf = row.get("drv_rollN_dnf_rate_5")
        drivers.append(DriverState(
            driver_id=row["driver_id"],
            pace=float(pace),
            dnf_rate=float(0.05 if pd.isna(dnf) else max(0.01, dnf)),
            start_compound="MEDIUM",
            # Single-stop baseline at 45% of race distance on hard.
            pit_laps=(int(0.45 * 57),),
            pit_compounds=("HARD",),
        ))
    return drivers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--laps", type=int, default=57)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES)
    grid = build_grid(df)
    last_row = df[df["round"] == df["round"].max()].iloc[0]
    config = RaceConfig(circuit_id=last_row["circuit_id"], total_laps=args.laps)
    print(f"Simulating {args.sims}× {last_row['race_name']} ({config.circuit_id}) "
          f"· {args.laps} laps · {len(grid)} drivers\n")

    mc = monte_carlo(grid, config, n_sims=args.sims, seed=args.seed)
    stats = summary(mc)
    rows = sorted(stats.items(), key=lambda kv: kv[1]["mean_finish"])
    print(f"{'driver':<20}  {'mean':>5}  {'P(win)':>6}  {'P(pod)':>6}  {'P(pts)':>6}")
    for drv, s in rows:
        print(f"  {drv:<18}  {s['mean_finish']:5.2f}  {s['p_win']:6.1%}  "
              f"{s['p_podium']:6.1%}  {s['p_points']:6.1%}")
    print(f"\nSC prob (per race): {mc['_sc_probability'][0]:.1%}")


if __name__ == "__main__":
    main()
