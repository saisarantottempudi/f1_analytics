"""Ergast / Jolpica F1 API client.

Ergast (ergast.com/mrd) was sunset end of 2024; the community-hosted Jolpica
mirror at api.jolpi.ca is the drop-in replacement. Interface is identical so
existing Ergast URLs just work.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100  # Jolpica hard-caps at 100
# Jolpica rate limit: 4 req/s burst, 500/hour. Sleeping 0.25s between calls
# keeps us inside both.
REQUEST_DELAY_S = 0.25


class ErgastError(RuntimeError):
    pass


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}/{path.lstrip('/')}.json"
    resp = requests.get(url, params=params or {}, timeout=30)
    if resp.status_code == 429:
        # Backoff and retry once.
        time.sleep(2)
        resp = requests.get(url, params=params or {}, timeout=30)
    if not resp.ok:
        raise ErgastError(f"{resp.status_code} from {url}: {resp.text[:200]}")
    return resp.json()["MRData"]


def _paginate(path: str) -> Iterator[dict[str, Any]]:
    """Yield each MRData page until the server reports no more rows."""
    offset = 0
    while True:
        mrdata = _get(path, {"limit": PAGE_LIMIT, "offset": offset})
        total = int(mrdata["total"])
        yield mrdata
        offset += PAGE_LIMIT
        if offset >= total:
            return
        time.sleep(REQUEST_DELAY_S)


def season_schedule(year: int) -> list[dict[str, Any]]:
    """Race calendar for a season — round, name, circuit, date."""
    out: list[dict[str, Any]] = []
    for page in _paginate(f"{year}"):
        for race in page["RaceTable"]["Races"]:
            out.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "circuit_name": race["Circuit"]["circuitName"],
                    "country": race["Circuit"]["Location"]["country"],
                    "locality": race["Circuit"]["Location"]["locality"],
                    "date": race["date"],
                }
            )
    return out


def season_results(year: int) -> list[dict[str, Any]]:
    """Per-driver race results for a season."""
    out: list[dict[str, Any]] = []
    for page in _paginate(f"{year}/results"):
        for race in page["RaceTable"]["Races"]:
            for r in race["Results"]:
                fl = r.get("FastestLap") or {}
                out.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "race_name": race["raceName"],
                        "circuit_id": race["Circuit"]["circuitId"],
                        "date": race["date"],
                        "driver_id": r["Driver"]["driverId"],
                        "driver_code": r["Driver"].get("code"),
                        "driver_name": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                        "constructor_id": r["Constructor"]["constructorId"],
                        "grid": int(r["grid"]),
                        "position": int(r["position"]) if r["position"].isdigit() else None,
                        "position_text": r["positionText"],
                        "points": float(r["points"]),
                        "laps": int(r["laps"]),
                        "status": r["status"],
                        "fastest_lap_rank": int(fl["rank"]) if fl.get("rank", "").isdigit() else None,
                        "fastest_lap_time": (fl.get("Time") or {}).get("time"),
                    }
                )
    return out


def season_qualifying(year: int) -> list[dict[str, Any]]:
    """Qualifying results (Q1/Q2/Q3) per driver per race. Available from 2003+."""
    out: list[dict[str, Any]] = []
    for page in _paginate(f"{year}/qualifying"):
        for race in page["RaceTable"]["Races"]:
            for q in race["QualifyingResults"]:
                out.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "date": race["date"],
                        "driver_id": q["Driver"]["driverId"],
                        "constructor_id": q["Constructor"]["constructorId"],
                        "position": int(q["position"]),
                        "q1": q.get("Q1"),
                        "q2": q.get("Q2"),
                        "q3": q.get("Q3"),
                    }
                )
    return out


def season_pitstops(year: int) -> list[dict[str, Any]]:
    """Pit stops per lap per driver. Ergast coverage starts 2012."""
    if year < 2012:
        return []
    # Pit stops must be requested per race — there's no season-wide endpoint.
    out: list[dict[str, Any]] = []
    schedule = season_schedule(year)
    for race in schedule:
        rnd = race["round"]
        try:
            mrdata = _get(f"{year}/{rnd}/pitstops", {"limit": PAGE_LIMIT})
        except ErgastError:
            continue
        races = mrdata["RaceTable"]["Races"]
        if not races:
            continue
        for stop in races[0].get("PitStops", []):
            out.append(
                {
                    "season": year,
                    "round": rnd,
                    "driver_id": stop["driverId"],
                    "stop": int(stop["stop"]),
                    "lap": int(stop["lap"]),
                    "time_of_day": stop["time"],
                    "duration_s": float(stop["duration"]) if stop.get("duration") else None,
                }
            )
        time.sleep(REQUEST_DELAY_S)
    return out
