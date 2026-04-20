"""POST /simulate — single race with lap-by-lap timeline."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api import state
from backend.api.predict import _driver_state_from_features
from backend.api.schemas import (
    FinalStandingRow, LapSnapshot, RaceEvent, SimulateRequest, SimulateResponse,
    StrategyEntry,
)
from backend.ml.monte_carlo import RaceConfig, simulate_race

router = APIRouter(tags=["simulate"])


def _apply_strategies(drivers, strategies: list[StrategyEntry]) -> None:
    """Override default strategies on the `drivers` list in place."""
    by_id = {d.driver_id: d for d in drivers}
    for s in strategies:
        if s.driver_id not in by_id:
            continue
        d = by_id[s.driver_id]
        d.start_compound = s.start_compound
        d.pit_laps = tuple(s.pit_laps)
        d.pit_compounds = tuple(s.pit_compounds)
        d.current_compound = s.start_compound


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    if not req.grid:
        raise HTTPException(400, "grid must contain at least one driver")

    grid_sorted = sorted(req.grid, key=lambda e: e.grid_position)
    drivers = [_driver_state_from_features(e, req.weather.rain_probability) for e in grid_sorted]
    _apply_strategies(drivers, req.strategies)

    config = RaceConfig(circuit_id=req.circuit_id, total_laps=req.total_laps)
    rng = np.random.default_rng(req.seed)
    res = simulate_race(drivers, config, rng, record_timeline=True)

    final = [
        FinalStandingRow(
            driver_id=drv,
            position=res.per_driver_position[drv],
            retired_lap=res.per_driver_retired_lap[drv],
        )
        for drv in res.finish_order
    ]
    timeline = [LapSnapshot(**snap) for snap in res.timeline]
    events = [
        RaceEvent(lap=e.lap, kind=e.kind, driver_id=e.driver_id, detail=e.detail)
        for e in res.events
    ]
    return SimulateResponse(
        circuit_id=req.circuit_id,
        total_laps=req.total_laps,
        laps_ran=res.laps_ran,
        final_standings=final,
        timeline=timeline,
        events=events,
    )
