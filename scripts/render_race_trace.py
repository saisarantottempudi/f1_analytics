"""Render a race-trace PNG from a single seeded Monte Carlo simulation.

Pulls the grid from the most recent race in the feature table, runs
`simulate_race` with a fixed seed, and writes
`notebooks/figures/06_race_trace.png`.

Used by the README to show what the simulator produces without asking the
reader to launch the UI.

Usage:
    python scripts/render_race_trace.py --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.api import state  # noqa: E402
from backend.api.predict import _driver_state_from_features  # noqa: E402
from backend.api.schemas import GridEntry  # noqa: E402
from backend.ml.monte_carlo import RaceConfig, simulate_race  # noqa: E402

OUT_PATH = ROOT / "notebooks" / "figures" / "06_race_trace.png"


def _latest_grid() -> tuple[str, int, list[GridEntry], str]:
    df = state.features()
    last = df[["season", "round"]].drop_duplicates().iloc[-1]
    sl = df[(df["season"] == last["season"]) & (df["round"] == last["round"])].copy()
    sl = sl[sl["grid"] > 0].sort_values("grid").head(20)
    label = f"{int(last['season'])} R{int(last['round']):02d} · {sl.iloc[0]['race_name']}"
    grid = [
        GridEntry(driver_id=r.driver_id, grid_position=int(r.grid),
                  constructor_id=r.constructor_id)
        for r in sl.itertuples()
    ]
    return str(sl.iloc[0]["circuit_id"]), 57, grid, label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    circuit_id, total_laps, grid, label = _latest_grid()
    drivers = [_driver_state_from_features(e, 0.0) for e in grid]
    cfg = RaceConfig(circuit_id=circuit_id, total_laps=total_laps)
    res = simulate_race(drivers, cfg, np.random.default_rng(args.seed),
                        record_timeline=True)

    # Build position-per-lap matrix.
    driver_ids = [e.driver_id for e in grid]
    pos_by_driver: dict[str, list[int | None]] = {d: [] for d in driver_ids}
    for snap in res.timeline:
        seen = set()
        for row in snap["standings"]:
            seen.add(row["driver_id"])
            pos_by_driver[row["driver_id"]].append(
                None if row["retired"] else row["position"]
            )
        for d in driver_ids:
            if d not in seen:
                pos_by_driver[d].append(None)

    laps = list(range(1, len(res.timeline) + 1))

    # Rank drivers by final finish position for legend ordering.
    final_order = sorted(
        driver_ids,
        key=lambda d: res.per_driver_position.get(d, 99),
    )

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
    cmap = plt.get_cmap("tab20")
    for i, drv in enumerate(final_order):
        y = pos_by_driver[drv]
        ax.plot(laps, y, marker="", linewidth=1.8, alpha=0.9,
                color=cmap(i % 20), label=drv)
    ax.invert_yaxis()
    ax.set_xlabel("lap")
    ax.set_ylabel("position")
    ax.set_title(f"Race trace — {label} (seed={args.seed})")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_yticks(range(1, len(driver_ids) + 1))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out.relative_to(ROOT)} "
          f"({len(driver_ids)} drivers, {len(laps)} laps)")


if __name__ == "__main__":
    main()
