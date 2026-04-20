"""End-to-end smoke test for the FastAPI layer.

Starts the app in-process with httpx.ASGITransport — no network / uvicorn needed —
then hits /predict, /simulate, and /h2h with realistic payloads and asserts the
responses look sane.

Usage:
    python scripts/smoke_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.api import state  # noqa: E402
from backend.main import app  # noqa: E402


def _grid_from_latest_race() -> tuple[str, list[dict]]:
    df = state.features()
    last = df[["season", "round"]].drop_duplicates().iloc[-1]
    sl = df[(df["season"] == last["season"]) & (df["round"] == last["round"])].copy()
    sl = sl.sort_values("grid").head(20)
    grid = [
        {"driver_id": r["driver_id"], "grid_position": int(r["grid"]),
         "constructor_id": r["constructor_id"]}
        for _, r in sl.iterrows() if r["grid"] > 0
    ]
    return str(sl.iloc[0]["circuit_id"]), grid


def main() -> None:
    circuit_id, grid = _grid_from_latest_race()
    client = TestClient(app)

    # /health ---
    r = client.get("/health")
    assert r.status_code == 200, r.text
    print("/health        ✅", r.json())

    # /predict ---
    r = client.post("/predict", json={
        "circuit_id": circuit_id,
        "total_laps": 55,
        "grid": grid,
        "weather": {"air_temp_c": 24.0, "rain_probability": 0.1},
        "n_sims": 200,
    })
    assert r.status_code == 200, r.text
    j = r.json()
    print(f"/predict       ✅  n_sims={j['n_sims']}  winner={j['winner_driver_id']}  "
          f"SC={j['sc_probability']:.0%}")
    top3 = sorted(j["drivers"], key=lambda d: -d["p_podium"])[:3]
    for d in top3:
        print(f"                · {d['driver_id']:<18} p_podium={d['p_podium']:.1%}")

    # /simulate ---
    r = client.post("/simulate", json={
        "circuit_id": circuit_id,
        "total_laps": 55,
        "grid": grid,
        "strategies": [],
        "weather": {"air_temp_c": 24.0, "rain_probability": 0.1},
        "seed": 7,
    })
    assert r.status_code == 200, r.text
    j = r.json()
    print(f"/simulate      ✅  laps_ran={j['laps_ran']}  "
          f"timeline_snapshots={len(j['timeline'])}  events={len(j['events'])}  "
          f"winner={j['final_standings'][0]['driver_id']}")

    # /h2h ---  Pick two drivers that share races.
    top_drivers = [d["driver_id"] for d in grid[:6]]
    r = client.post("/h2h", json={
        "driver_a": top_drivers[0],
        "driver_b": top_drivers[1],
    })
    assert r.status_code == 200, r.text
    j = r.json()
    print(f"/h2h           ✅  {j['driver_a']} vs {j['driver_b']}  "
          f"shared={j['shared_races']}  winner={j['overall_winner']}  "
          f"edge={j['overall_edge_pct']:.1f}%")
    for s in j["sections"][:4]:
        print(f"                · {s['label']:<40} A={s['a_value']}  B={s['b_value']}  → {s['winner']}")

    print("\nAll endpoints OK.")


if __name__ == "__main__":
    main()
