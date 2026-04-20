"""POST /h2h — two-driver comparison over the feature/results history."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.api import state
from backend.api.explain import explain
from backend.api.schemas import H2HRequest, H2HResponse, H2HSection
from rag.prompts import h2h_prompt

router = APIRouter(tags=["h2h"])


def _winner(a: float | None, b: float | None, smaller_is_better: bool) -> str | None:
    if a is None or b is None or a != a or b != b:
        return None
    if a == b:
        return "TIE"
    if smaller_is_better:
        return "A" if a < b else "B"
    return "A" if a > b else "B"


@router.post("/h2h", response_model=H2HResponse)
def head_to_head(
    req: H2HRequest,
    explain_flag: bool = Query(False, alias="explain"),
    llm: bool = Query(False),
) -> H2HResponse:
    df = state.features()
    if req.driver_a == req.driver_b:
        raise HTTPException(400, "driver_a and driver_b must differ")

    if req.season_from is not None:
        df = df[df["season"] >= req.season_from]
    if req.season_to is not None:
        df = df[df["season"] <= req.season_to]
    if req.circuit_id is not None:
        df = df[df["circuit_id"] == req.circuit_id]

    a = df[df["driver_id"] == req.driver_a]
    b = df[df["driver_id"] == req.driver_b]
    if a.empty or b.empty:
        raise HTTPException(404, "one or both drivers have no rows in the filtered window")

    keys = ["season", "round"]
    shared = a.merge(b, on=keys, suffixes=("_a", "_b"))
    shared_n = len(shared)

    sections: list[H2HSection] = []

    def add(label: str, av, bv, smaller_better: bool):
        av = float(av) if av == av else None
        bv = float(bv) if bv == bv else None
        sections.append(H2HSection(
            label=label, a_value=av, b_value=bv,
            winner=_winner(av, bv, smaller_better),
        ))

    # Qualifying (grid) — smaller is better.
    add("Avg grid position", a["grid"].mean(), b["grid"].mean(), True)
    if shared_n:
        a_quali = (shared["grid_a"] < shared["grid_b"]).sum()
        b_quali = (shared["grid_b"] < shared["grid_a"]).sum()
        add("Head-to-head quali wins (shared races)", a_quali, b_quali, False)

    # Race finish — smaller is better.
    add("Avg finish position",
        a[a["is_finish"] == 1]["position"].mean(),
        b[b["is_finish"] == 1]["position"].mean(),
        True)

    # Positions gained/lost (grid − finish — larger is better).
    a_gained = (a["grid"] - a["position"]).mean()
    b_gained = (b["grid"] - b["position"]).mean()
    add("Avg positions gained on race day", a_gained, b_gained, False)

    # Consistency (std of finish — smaller is better).
    a_cons = a[a["is_finish"] == 1]["position"].std()
    b_cons = b[b["is_finish"] == 1]["position"].std()
    add("Finish-position std (lower = more consistent)", a_cons, b_cons, True)

    # DNF rate — smaller is better.
    add("DNF rate",
        1.0 - a["is_finish"].mean(),
        1.0 - b["is_finish"].mean(),
        True)

    # Wins / podiums — larger is better.
    add("Wins (P1 count)",
        (a["position"] == 1).sum(),
        (b["position"] == 1).sum(),
        False)
    add("Podiums (P≤3 count)",
        (a["position"] <= 3).sum(),
        (b["position"] <= 3).sum(),
        False)

    a_points = sum(1 for s in sections if s.winner == "A")
    b_points = sum(1 for s in sections if s.winner == "B")
    total_decided = a_points + b_points
    if total_decided == 0:
        overall = "TIE"
        edge_pct = 0.0
    elif a_points > b_points:
        overall = "A"
        edge_pct = 100.0 * (a_points - b_points) / total_decided
    elif b_points > a_points:
        overall = "B"
        edge_pct = 100.0 * (b_points - a_points) / total_decided
    else:
        overall = "TIE"
        edge_pct = 0.0

    seasons = (int(df["season"].min()), int(df["season"].max())) if not df.empty else None
    resp = H2HResponse(
        driver_a=req.driver_a,
        driver_b=req.driver_b,
        season_range=seasons,
        shared_races=shared_n,
        sections=sections,
        overall_winner=overall,
        overall_edge_pct=edge_pct,
    )

    if explain_flag:
        h2h_dict = resp.model_dump()
        retrieval_filters: dict = {}
        if req.circuit_id:
            retrieval_filters["circuit_id"] = req.circuit_id
        if req.season_from is not None:
            retrieval_filters["season_from"] = req.season_from
        if req.season_to is not None:
            retrieval_filters["season_to"] = req.season_to
        narrative, source = explain(
            retrieval_query=f"{req.driver_a} vs {req.driver_b} battle quali race",
            retrieval_filters=retrieval_filters,
            build_prompt=lambda retrieved: h2h_prompt(h2h_dict, retrieved),
            use_llm=llm,
        )
        resp.narrative = narrative
        resp.narrative_source = source
    return resp
