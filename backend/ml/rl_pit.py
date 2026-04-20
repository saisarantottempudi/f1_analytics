"""Tabular Q-learning pit-strategy agent.

The agent decides, each lap, whether to stay out or pit for one of the three
compounds. Reward is `-(finish_position)` at race end; intermediate steps get 0.

We use a tabular Q-table over a small discretised state space so this trains
on a laptop in seconds. If the reward landscape turns out too sparse we can
upgrade to DQN later — the interface is compatible.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.ml.monte_carlo import (
    Compound, DriverState, RaceConfig, simulate_race,
)

ROOT = Path(__file__).resolve().parents[2]
Q_PATH = ROOT / "models" / "rl_pit_q.json"

ACTIONS: tuple[str, ...] = ("STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD")


@dataclass(frozen=True)
class State:
    phase: int         # Race-phase quartile (0..3)
    tyre_age: int      # Bucket (0..3)
    pace_rank: int     # Current position bucket (0..3)
    laps_left: int     # Bucket (0..2)

    def key(self) -> str:
        return f"{self.phase}-{self.tyre_age}-{self.pace_rank}-{self.laps_left}"


def _bucket_tyre_age(age: int) -> int:
    if age < 8:  return 0
    if age < 16: return 1
    if age < 24: return 2
    return 3


def _bucket_pace_rank(pos: int, field_size: int) -> int:
    frac = pos / max(field_size, 1)
    if frac < 0.25: return 0
    if frac < 0.50: return 1
    if frac < 0.75: return 2
    return 3


def _bucket_laps_left(left: int, total: int) -> int:
    frac = left / max(total, 1)
    if frac > 0.5:  return 0
    if frac > 0.15: return 1
    return 2


def _current_state(lap: int, agent: DriverState, drivers: list[DriverState],
                   config: RaceConfig) -> State:
    # Derive running order from partial cumulative time.
    sorted_drv = sorted(drivers, key=lambda d: (d.retired, d.cum_time))
    pos = sorted_drv.index(agent) + 1
    return State(
        phase=min(3, int(4 * lap / max(config.total_laps, 1))),
        tyre_age=_bucket_tyre_age(agent.tyre_age),
        pace_rank=_bucket_pace_rank(pos, len(drivers)),
        laps_left=_bucket_laps_left(config.total_laps - lap, config.total_laps),
    )


class QTable:
    def __init__(self) -> None:
        self.q: dict[str, np.ndarray] = {}

    def get(self, s: State) -> np.ndarray:
        k = s.key()
        if k not in self.q:
            self.q[k] = np.zeros(len(ACTIONS), dtype=np.float32)
        return self.q[k]

    def save(self, path: Path = Q_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {k: v.tolist() for k, v in self.q.items()}, indent=2))

    @classmethod
    def load(cls, path: Path = Q_PATH) -> "QTable":
        obj = cls()
        if path.exists():
            blob = json.loads(path.read_text())
            obj.q = {k: np.array(v, dtype=np.float32) for k, v in blob.items()}
        return obj


def epsilon_greedy(q_row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(len(ACTIONS)))
    return int(q_row.argmax())


def _run_episode(
    agent_template: DriverState,
    opponents: list[DriverState],
    config: RaceConfig,
    qt: QTable,
    eps: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, int]], float]:
    """Run one race with the agent choosing pit actions per lap; update nothing."""
    agent = copy.deepcopy(agent_template)
    opps = [copy.deepcopy(o) for o in opponents]
    drivers = [agent, *opps]

    # Reset state.
    for d in drivers:
        d.cum_time = 0.0
        d.tyre_age = 0
        d.current_compound = d.start_compound
        d.retired = False

    trajectory: list[tuple[str, int]] = []
    sc_remaining = 0
    vsc_remaining = 0
    from backend.ml.monte_carlo import (
        SC_LAP_PROB, VSC_LAP_PROB, SC_DURATION, VSC_DURATION,
        _compound_step_cost, FUEL_GAIN_S, PIT_LOSS_S, _strategy_for_lap,
    )

    for lap in range(1, config.total_laps + 1):
        if sc_remaining == 0 and vsc_remaining == 0:
            r = rng.random()
            if r < SC_LAP_PROB:
                sc_remaining = SC_DURATION
            elif r < SC_LAP_PROB + VSC_LAP_PROB:
                vsc_remaining = VSC_DURATION

        sc_multiplier = 1.0
        if sc_remaining > 0:
            sc_multiplier, sc_remaining = 1.35, sc_remaining - 1
        elif vsc_remaining > 0:
            sc_multiplier, vsc_remaining = 1.15, vsc_remaining - 1

        # Agent chooses.
        s = _current_state(lap, agent, drivers, config)
        action_idx = epsilon_greedy(qt.get(s), eps, rng)
        action = ACTIONS[action_idx]
        trajectory.append((s.key(), action_idx))

        # Apply agent action.
        agent_pit_cost = 0.0
        if action != "STAY" and not agent.retired:
            comp: Compound = action.split("_")[1]  # type: ignore[assignment]
            agent.current_compound = comp
            agent.tyre_age = 0
            agent_pit_cost = PIT_LOSS_S

        # Step every driver (agent + opponents).
        for d in drivers:
            if d.retired:
                continue
            if d is agent:
                pit_cost = agent_pit_cost
            else:
                nc = _strategy_for_lap(d, lap)
                pit_cost = PIT_LOSS_S if nc is not None else 0.0
                if nc is not None:
                    d.current_compound = nc
                    d.tyre_age = 0

            base = config.base_lap_time_s + d.pace
            tyre_cost = _compound_step_cost(d.current_compound, d.tyre_age, d.tyre_mgmt)
            lap_time = (base + tyre_cost - FUEL_GAIN_S * lap
                        + rng.normal(0, 0.15)) * sc_multiplier + pit_cost
            d.cum_time += lap_time
            d.tyre_age += 1

    # Reward = -finish_position.
    ranked = sorted(drivers, key=lambda d: (d.retired, d.cum_time))
    agent_pos = ranked.index(agent) + 1
    reward = -float(agent_pos)
    return trajectory, reward


def train(
    agent_template: DriverState,
    opponents: list[DriverState],
    config: RaceConfig,
    n_episodes: int = 400,
    lr: float = 0.3,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    seed: int = 0,
) -> QTable:
    """Train a Q-table via Monte Carlo returns on synthetic races."""
    rng = np.random.default_rng(seed)
    qt = QTable()
    for ep in range(n_episodes):
        eps = eps_end + (eps_start - eps_end) * (1 - ep / max(n_episodes - 1, 1))
        traj, reward = _run_episode(
            agent_template, opponents, config, qt, eps, rng,
        )
        # Monte Carlo update — whole-episode return applied to every visited (s, a).
        for key, a in traj:
            row = qt.q.setdefault(key, np.zeros(len(ACTIONS), dtype=np.float32))
            row[a] += lr * (reward - row[a])
    return qt


def best_action(qt: QTable, s: State) -> str:
    return ACTIONS[int(qt.get(s).argmax())]
