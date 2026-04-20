"""GET /meta — schedule, driver roster, and circuit list for the frontend."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api import state

router = APIRouter(tags=["meta"])


@router.get("/meta")
def meta() -> dict:
    df = state.features()
    sched = state.schedule()

    # Schedule rows, newest-first.
    schedule_rows = (
        sched.sort_values(["season", "round"], ascending=[False, True])
             .to_dict(orient="records")
        if not sched.empty else []
    )
    # Parquet dates / numpy types → plain Python for JSON.
    for r in schedule_rows:
        r["season"] = int(r["season"])
        r["round"] = int(r["round"])
        r["date"] = str(r["date"])

    # Drivers — take last-seen constructor as the current team.
    latest = (
        df.sort_values(["season", "round"])
          .groupby("driver_id", as_index=False)
          .tail(1)
    )
    drivers = [
        {
            "driver_id": r.driver_id,
            "driver_name": r.driver_name,
            "constructor_id": r.constructor_id,
        }
        for r in latest.itertuples()
    ]
    drivers.sort(key=lambda d: d["driver_name"])

    circuits = sorted(df["circuit_id"].unique().tolist())
    seasons = sorted(df["season"].unique().tolist())

    return {
        "seasons": [int(s) for s in seasons],
        "circuits": circuits,
        "drivers": drivers,
        "schedule": schedule_rows,
    }
