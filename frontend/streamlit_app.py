"""🏎️ F1 Analytics AI — Home Dashboard (Race Control Center).

Pick a race + grid + weather here. The selection is stored in
`st.session_state` and consumed by the other three pages.

Run alongside the API:
    uvicorn backend.main:app --reload
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from frontend.api_client import health, meta

st.set_page_config(page_title="F1 Analytics AI", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Analytics AI — Race Control Center")
st.caption("Prediction · full-race simulation · driver head-to-head, all backed "
           "by FastF1 + Ergast data and the local FastAPI backend.")

# ---------- API health ----------

h = health()
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("API status", "online" if h else "offline")
with col_b:
    st.metric("RAG index", h.get("rag_index_ready", "?") if h else "—")
with col_c:
    st.metric("LLM key", h.get("llm_key_present", "?") if h else "—")

if not h:
    st.error("Backend not reachable on http://127.0.0.1:8000. "
             "Start it with `uvicorn backend.main:app --reload` and refresh.")
    st.stop()

m = meta()
if not m:
    st.stop()

st.divider()

# ---------- Race + grid selection ----------

st.subheader("🏁 Pick a race")

sched = m["schedule"]
if not sched:
    st.warning("No schedule found — run `scripts/ingest_ergast.py` first.")
    st.stop()

race_labels = [f"{r['season']} · R{r['round']:02d} · {r['race_name']}" for r in sched]
sel_ix = st.selectbox("Race", range(len(race_labels)),
                      format_func=lambda i: race_labels[i], index=0)
race = sched[sel_ix]
circuit_id = race["circuit_id"]

col1, col2, col3 = st.columns(3)
col1.metric("Circuit", circuit_id)
col2.metric("Country", race.get("country", "—"))
col3.metric("Date", race["date"])

# ---------- Grid builder ----------

st.subheader("👥 Grid")

drivers = m["drivers"]
driver_ids = [d["driver_id"] for d in drivers]
driver_labels = {d["driver_id"]: f"{d['driver_name']} ({d['constructor_id']})"
                 for d in drivers}

default_top = driver_ids[: min(20, len(driver_ids))]
sel = st.multiselect(
    "Drivers on the grid (in selected starting order)",
    options=driver_ids,
    default=default_top,
    format_func=lambda d: driver_labels.get(d, d),
    max_selections=20,
)

if len(sel) < 2:
    st.warning("Pick at least two drivers.")
    st.stop()

st.caption("Order above = starting grid (P1 first). Drag to re-order isn't built-in; "
           "remove and re-add in the right sequence to change grid positions.")

grid = [
    {
        "driver_id": d,
        "grid_position": i + 1,
        "constructor_id": next(
            (dr["constructor_id"] for dr in drivers if dr["driver_id"] == d), None
        ),
    }
    for i, d in enumerate(sel)
]

# ---------- Weather + sims ----------

st.subheader("🌤️ Weather & simulation settings")

cw1, cw2, cw3 = st.columns(3)
air_temp = cw1.slider("Air temperature (°C)", 5.0, 45.0, 24.0, 0.5)
rain_prob = cw2.slider("Rain probability", 0.0, 1.0, 0.10, 0.05)
total_laps = cw3.number_input("Total laps", 30, 80, 57, 1)

n_sims = st.slider("Monte Carlo runs (Prediction page)", 100, 2000, 500, 50)

# ---------- Persist selection ----------

st.session_state["circuit_id"] = circuit_id
st.session_state["race_label"] = race_labels[sel_ix]
st.session_state["total_laps"] = int(total_laps)
st.session_state["grid"] = grid
st.session_state["weather"] = {
    "air_temp_c": float(air_temp),
    "rain_probability": float(rain_prob),
}
st.session_state["n_sims"] = int(n_sims)
st.session_state["drivers_map"] = driver_labels

st.divider()
st.success(f"✅ Selection saved — {len(grid)} drivers, {total_laps} laps, "
           f"rain {rain_prob:.0%}. Use the sidebar to open Prediction, "
           f"Simulation, or Head-to-Head.")
