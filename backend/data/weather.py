"""Weather lookup with OpenWeatherMap primary + FastF1 session fallback.

- Historical race weather: pulled from FastF1's per-session `weather_data`
  (air/track temp, humidity, wind, rainfall flag). Works offline after cache.
- Future race weather: OpenWeather 5-day forecast if OPENWEATHER_API_KEY is set;
  otherwise a neutral stub so downstream sims keep running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

OWM_BASE = "https://api.openweathermap.org/data/2.5"


@dataclass
class Weather:
    air_temp_c: float
    track_temp_c: float | None
    humidity_pct: float
    wind_kph: float
    rain_probability: float  # 0.0 - 1.0
    source: str  # "fastf1" | "openweather" | "stub"


def _stub() -> Weather:
    # Neutral dry-race defaults — used when no key and no historical data.
    return Weather(25.0, 35.0, 55.0, 10.0, 0.05, "stub")


def from_fastf1_session(session: Any) -> Weather:
    """Derive mean race weather from a loaded FastF1 session object."""
    wdf = session.weather_data
    if wdf is None or wdf.empty:
        return _stub()
    rain_flag = bool(wdf["Rainfall"].any()) if "Rainfall" in wdf else False
    return Weather(
        air_temp_c=float(wdf["AirTemp"].mean()),
        track_temp_c=float(wdf["TrackTemp"].mean()) if "TrackTemp" in wdf else None,
        humidity_pct=float(wdf["Humidity"].mean()),
        wind_kph=float(wdf["WindSpeed"].mean()) * 3.6,  # m/s → kph
        rain_probability=1.0 if rain_flag else 0.0,
        source="fastf1",
    )


def forecast(lat: float, lon: float, when: datetime) -> Weather:
    """OpenWeather 5-day/3-hour forecast; picks the slot nearest `when`."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return _stub()
    resp = requests.get(
        f"{OWM_BASE}/forecast",
        params={"lat": lat, "lon": lon, "appid": key, "units": "metric"},
        timeout=15,
    )
    if not resp.ok:
        return _stub()
    slots = resp.json().get("list", [])
    if not slots:
        return _stub()
    target_ts = when.timestamp()
    best = min(slots, key=lambda s: abs(s["dt"] - target_ts))
    main = best["main"]
    wind = best.get("wind", {})
    rain_3h = (best.get("rain") or {}).get("3h", 0.0)
    return Weather(
        air_temp_c=float(main["temp"]),
        track_temp_c=None,
        humidity_pct=float(main["humidity"]),
        wind_kph=float(wind.get("speed", 0.0)) * 3.6,
        rain_probability=min(1.0, rain_3h / 2.0),  # rough mm→prob mapping
        source="openweather",
    )
