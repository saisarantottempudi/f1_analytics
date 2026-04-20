"""Pydantic request/response schemas for the three API endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Compound = Literal["SOFT", "MEDIUM", "HARD"]


# ---------- shared ----------

class GridEntry(BaseModel):
    driver_id: str
    grid_position: int = Field(ge=1, le=22)
    constructor_id: str | None = None


class WeatherInput(BaseModel):
    air_temp_c: float = 25.0
    rain_probability: float = Field(0.0, ge=0.0, le=1.0)


class StrategyEntry(BaseModel):
    driver_id: str
    start_compound: Compound = "MEDIUM"
    pit_laps: list[int] = Field(default_factory=list)
    pit_compounds: list[Compound] = Field(default_factory=list)


# ---------- /predict ----------

class PredictRequest(BaseModel):
    circuit_id: str
    total_laps: int = 57
    grid: list[GridEntry]
    weather: WeatherInput = WeatherInput()
    n_sims: int = Field(1000, ge=50, le=5000)


class DriverPrediction(BaseModel):
    driver_id: str
    grid_position: int
    mean_finish: float
    p_win: float
    p_podium: float
    p_points: float


class PredictResponse(BaseModel):
    circuit_id: str
    n_sims: int
    drivers: list[DriverPrediction]
    pole_driver_id: str
    winner_driver_id: str
    sc_probability: float


# ---------- /simulate ----------

class SimulateRequest(BaseModel):
    circuit_id: str
    total_laps: int = 57
    grid: list[GridEntry]
    strategies: list[StrategyEntry] = Field(default_factory=list)
    weather: WeatherInput = WeatherInput()
    seed: int | None = None


class LapStanding(BaseModel):
    driver_id: str
    position: int
    gap_s: float | None
    retired: bool
    compound: Compound
    tyre_age: int


class LapSnapshot(BaseModel):
    lap: int
    standings: list[LapStanding]


class RaceEvent(BaseModel):
    lap: int
    kind: Literal["SC", "VSC", "DNF", "PIT", "OVERTAKE"]
    driver_id: str | None = None
    detail: str = ""


class FinalStandingRow(BaseModel):
    driver_id: str
    position: int
    retired_lap: int | None = None


class SimulateResponse(BaseModel):
    circuit_id: str
    total_laps: int
    laps_ran: int
    final_standings: list[FinalStandingRow]
    timeline: list[LapSnapshot]
    events: list[RaceEvent]


# ---------- /h2h ----------

class H2HRequest(BaseModel):
    driver_a: str
    driver_b: str
    season_from: int | None = None
    season_to: int | None = None
    circuit_id: str | None = None


class H2HSection(BaseModel):
    label: str
    a_value: float | None
    b_value: float | None
    winner: Literal["A", "B", "TIE"] | None = None


class H2HResponse(BaseModel):
    driver_a: str
    driver_b: str
    season_range: tuple[int, int] | None
    shared_races: int
    sections: list[H2HSection]
    overall_winner: Literal["A", "B", "TIE"]
    overall_edge_pct: float
