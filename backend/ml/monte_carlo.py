"""Monte Carlo race simulator.

Drives a lap-by-lap race with:
    - Per-compound tyre degradation (linear model, data-driven defaults)
    - Fuel burn-off (small lap-time gain per lap)
    - Pit stops with a fixed time loss
    - Safety Car / VSC triggers (Poisson-like per-lap probability)
    - Driver pace + race noise from a driver-performance prior
    - Simple DNF model

Scope: "realistic-enough" — not a lap-simulator replacing Motorsport Manager.
It produces plausible finish-position distributions in ~50 ms per run so we
can Monte Carlo 1 000 sims for probabilistic predictions in the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Compound = Literal["SOFT", "MEDIUM", "HARD"]

# Tyre params: (base lap-time offset vs reference, deg seconds per lap).
# Reference is MEDIUM compound at age 0.
TYRE_PARAMS: dict[Compound, tuple[float, float]] = {
    "SOFT":   (-0.50, 0.08),
    "MEDIUM": ( 0.00, 0.04),
    "HARD":   ( 0.35, 0.02),
}

PIT_LOSS_S = 22.0       # Fixed pit time loss (stationary + slow zone).
FUEL_GAIN_S = 0.03      # Per-lap lap-time reduction as fuel burns off.
SC_LAP_PROB = 0.015     # Full safety car trigger / lap (~30% for a 60-lap race).
VSC_LAP_PROB = 0.025    # Virtual safety car trigger / lap.
SC_DURATION = 5
VSC_DURATION = 3
OVERTAKE_BASE_P = 0.25  # Base overtake probability when pace delta + gap favour it.


@dataclass
class DriverState:
    driver_id: str
    pace: float                  # Seconds per lap relative to the fastest car.
    tyre_mgmt: float = 1.0       # Multiplier on tyre deg: 0.8 = great, 1.2 = rough.
    dnf_rate: float = 0.03       # Per-race DNF probability.
    start_compound: Compound = "MEDIUM"
    pit_laps: tuple[int, ...] = ()  # Fixed strategy; RL agent overrides this.
    pit_compounds: tuple[Compound, ...] = ()

    # Mutable per-sim state.
    cum_time: float = 0.0
    current_compound: Compound = field(init=False)
    tyre_age: int = 0
    retired: bool = False
    retired_lap: int | None = None
    position: int = 0

    def __post_init__(self) -> None:
        self.current_compound = self.start_compound


@dataclass
class RaceConfig:
    circuit_id: str
    total_laps: int = 57
    base_lap_time_s: float = 90.0      # Clean-air reference lap.
    overtake_difficulty: float = 0.5   # 0 = Monza, 1 = Monaco.


@dataclass
class LapEvent:
    lap: int
    kind: Literal["SC", "VSC", "DNF", "PIT", "OVERTAKE"]
    driver_id: str | None = None
    detail: str = ""


@dataclass
class SimResult:
    finish_order: list[str]
    per_driver_position: dict[str, int]
    per_driver_retired_lap: dict[str, int | None]
    events: list[LapEvent]
    laps_ran: int


def _compound_step_cost(c: Compound, age: int, tyre_mgmt: float) -> float:
    offset, deg = TYRE_PARAMS[c]
    return offset + deg * age * tyre_mgmt


def _strategy_for_lap(driver: DriverState, lap: int) -> Compound | None:
    """Return new compound if this driver is pitting on `lap`, else None."""
    for pit_lap, compound in zip(driver.pit_laps, driver.pit_compounds):
        if pit_lap == lap:
            return compound
    return None


def simulate_race(
    drivers: list[DriverState],
    config: RaceConfig,
    rng: np.random.Generator | None = None,
) -> SimResult:
    rng = rng or np.random.default_rng()
    events: list[LapEvent] = []
    sc_remaining = 0
    vsc_remaining = 0

    for d in drivers:
        d.cum_time = 0.0
        d.tyre_age = 0
        d.current_compound = d.start_compound
        d.retired = False
        d.retired_lap = None

    # Uniform random DNF lap for each driver (if they DNF at all).
    dnf_lap: dict[str, int | None] = {}
    for d in drivers:
        dnf_lap[d.driver_id] = (
            int(rng.integers(5, config.total_laps))
            if rng.random() < d.dnf_rate else None
        )

    laps_ran = config.total_laps
    for lap in range(1, config.total_laps + 1):
        # Safety-car sampling.
        if sc_remaining == 0 and vsc_remaining == 0:
            r = rng.random()
            if r < SC_LAP_PROB:
                sc_remaining = SC_DURATION
                events.append(LapEvent(lap, "SC"))
            elif r < SC_LAP_PROB + VSC_LAP_PROB:
                vsc_remaining = VSC_DURATION
                events.append(LapEvent(lap, "VSC"))

        sc_multiplier = 1.0
        if sc_remaining > 0:
            sc_multiplier = 1.35       # SC slow pace.
            sc_remaining -= 1
        elif vsc_remaining > 0:
            sc_multiplier = 1.15       # VSC slow pace.
            vsc_remaining -= 1

        for d in drivers:
            if d.retired:
                continue

            # DNF check.
            if dnf_lap[d.driver_id] == lap:
                d.retired = True
                d.retired_lap = lap
                events.append(LapEvent(lap, "DNF", d.driver_id))
                continue

            # Pit stop?
            new_comp = _strategy_for_lap(d, lap)
            pit_cost = 0.0
            if new_comp is not None:
                pit_cost = PIT_LOSS_S
                d.current_compound = new_comp
                d.tyre_age = 0
                events.append(LapEvent(lap, "PIT", d.driver_id, detail=new_comp))

            base = config.base_lap_time_s + d.pace
            tyre_cost = _compound_step_cost(d.current_compound, d.tyre_age, d.tyre_mgmt)
            fuel_gain = FUEL_GAIN_S * lap
            noise = rng.normal(0.0, 0.15)
            lap_time = (base + tyre_cost - fuel_gain + noise) * sc_multiplier + pit_cost
            d.cum_time += lap_time
            d.tyre_age += 1

        # Early finish if all retired except one (rare).
        alive = [d for d in drivers if not d.retired]
        if len(alive) <= 1:
            laps_ran = lap
            break

    order = sorted(drivers, key=lambda d: (d.retired, d.cum_time))
    finish_order = [d.driver_id for d in order]
    per_driver_pos = {d.driver_id: i + 1 for i, d in enumerate(order)}
    for d in drivers:
        d.position = per_driver_pos[d.driver_id]

    return SimResult(
        finish_order=finish_order,
        per_driver_position=per_driver_pos,
        per_driver_retired_lap={d.driver_id: d.retired_lap for d in drivers},
        events=events,
        laps_ran=laps_ran,
    )


def monte_carlo(
    drivers_template: list[DriverState],
    config: RaceConfig,
    n_sims: int = 1000,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Run `n_sims` races; return per-driver position distribution (length n_sims)."""
    import copy
    rng = np.random.default_rng(seed)
    results: dict[str, list[int]] = {d.driver_id: [] for d in drivers_template}
    sc_counts = 0
    for _ in range(n_sims):
        drivers = [copy.deepcopy(d) for d in drivers_template]
        res = simulate_race(drivers, config, rng)
        for drv, pos in res.per_driver_position.items():
            results[drv].append(pos)
        sc_counts += sum(1 for e in res.events if e.kind == "SC")

    out = {drv: np.array(pos_list) for drv, pos_list in results.items()}
    out["_sc_probability"] = np.array([sc_counts / n_sims])
    return out


def summary(monte: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Collapse a Monte-Carlo run into human-readable stats per driver."""
    out: dict[str, dict[str, float]] = {}
    for drv, arr in monte.items():
        if drv.startswith("_"):
            continue
        out[drv] = {
            "mean_finish": float(arr.mean()),
            "p_win": float((arr == 1).mean()),
            "p_podium": float((arr <= 3).mean()),
            "p_points": float((arr <= 10).mean()),
        }
    return out
