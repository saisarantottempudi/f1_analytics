"""POST /predict — Monte-Carlo-backed race prediction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.api import state
from backend.api.explain import explain
from backend.api.schemas import (
    DriverPrediction, GridEntry, PredictRequest, PredictResponse,
)
from backend.ml.monte_carlo import (
    DriverState, RaceConfig, monte_carlo, summary,
)
from rag.prompts import prediction_prompt

router = APIRouter(tags=["predict"])


def _driver_state_from_features(entry: GridEntry, weather_rain: float) -> DriverState:
    row = state.latest_driver_row(entry.driver_id)
    if row is None:
        return DriverState(
            driver_id=entry.driver_id, pace=1.5, dnf_rate=0.05,
            start_compound="MEDIUM",
            pit_laps=(int(0.45 * 57),), pit_compounds=("HARD",),
        )
    form = row.get("drv_rollN_avg_finish_5")
    dnf = row.get("drv_rollN_dnf_rate_5")
    pace = 0.2 * ((form if form == form else 10) - 1)  # NaN-safe
    rain_bump = 0.03 * weather_rain
    return DriverState(
        driver_id=entry.driver_id,
        pace=float(pace),
        tyre_mgmt=1.0,
        dnf_rate=float(max(0.01, (dnf if dnf == dnf else 0.05) + rain_bump)),
        start_compound="MEDIUM",
        pit_laps=(int(0.45 * 57),),
        pit_compounds=("HARD",),
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    explain_flag: bool = Query(False, alias="explain"),
    llm: bool = Query(False),
) -> PredictResponse:
    if not req.grid:
        raise HTTPException(400, "grid must contain at least one driver")

    grid_sorted = sorted(req.grid, key=lambda e: e.grid_position)
    drivers = [_driver_state_from_features(e, req.weather.rain_probability) for e in grid_sorted]
    config = RaceConfig(circuit_id=req.circuit_id, total_laps=req.total_laps)

    mc = monte_carlo(drivers, config, n_sims=req.n_sims, seed=0)
    stats = summary(mc)

    predictions = [
        DriverPrediction(
            driver_id=e.driver_id,
            grid_position=e.grid_position,
            mean_finish=stats[e.driver_id]["mean_finish"],
            p_win=stats[e.driver_id]["p_win"],
            p_podium=stats[e.driver_id]["p_podium"],
            p_points=stats[e.driver_id]["p_points"],
        )
        for e in grid_sorted
    ]
    winner = max(predictions, key=lambda p: p.p_win).driver_id
    pole = grid_sorted[0].driver_id

    resp = PredictResponse(
        circuit_id=req.circuit_id,
        n_sims=req.n_sims,
        drivers=predictions,
        pole_driver_id=pole,
        winner_driver_id=winner,
        sc_probability=float(mc["_sc_probability"][0]),
    )

    if explain_flag:
        ml_dict = resp.model_dump()
        narrative, source = explain(
            retrieval_query=f"race at {req.circuit_id} historical pole winner podium",
            retrieval_filters={"circuit_id": req.circuit_id},
            build_prompt=lambda retrieved: prediction_prompt(ml_dict, retrieved),
            use_llm=llm,
        )
        resp.narrative = narrative
        resp.narrative_source = source
    return resp
