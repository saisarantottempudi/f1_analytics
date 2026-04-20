"""Train the tabular Q-learning pit-strategy agent against synthetic opponents.

Usage:
    python scripts/train_rl_pit.py
    python scripts/train_rl_pit.py --episodes 1000 --lr 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.monte_carlo import DriverState, RaceConfig  # noqa: E402
from backend.ml.rl_pit import ACTIONS, State, best_action, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.3)
    args = ap.parse_args()

    # Agent: mid-pack car. Opponents: fixed 1-stop strategies at varied pits.
    agent = DriverState(
        driver_id="AGENT", pace=0.3, dnf_rate=0.03,
        start_compound="MEDIUM",
    )
    opponents = []
    for i in range(9):
        opponents.append(DriverState(
            driver_id=f"OPP{i}",
            pace=0.05 * i,
            dnf_rate=0.03,
            start_compound="MEDIUM" if i % 2 else "SOFT",
            pit_laps=(20 + (i % 5) * 3,),
            pit_compounds=("HARD",),
        ))

    config = RaceConfig(circuit_id="generic", total_laps=57)

    print(f"Training {args.episodes} episodes…")
    qt = train(agent, opponents, config,
               n_episodes=args.episodes, lr=args.lr, seed=0)
    path = Path(__file__).resolve().parents[1] / "models" / "rl_pit_q.json"
    qt.save(path)
    print(f"Saved Q-table with {len(qt.q)} states → {path.relative_to(Path(__file__).resolve().parents[1])}")

    # Quick policy probe — what does the agent do mid-race on old tyres?
    print("\nPolicy spot-checks:")
    probes = [
        State(phase=1, tyre_age=2, pace_rank=2, laps_left=1),
        State(phase=2, tyre_age=3, pace_rank=1, laps_left=1),
        State(phase=3, tyre_age=1, pace_rank=0, laps_left=2),
    ]
    for s in probes:
        print(f"  {s.key():<16} → {best_action(qt, s)}")


if __name__ == "__main__":
    main()
