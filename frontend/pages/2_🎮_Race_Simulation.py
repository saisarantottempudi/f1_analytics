"""🎮 Race Simulation — lap-by-lap leaderboard, tyre strategy, replay, events."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import simulate

st.set_page_config(page_title="Race Simulation · F1 Analytics AI",
                   page_icon="🎮", layout="wide")
st.title("🎮 Race Simulation")

if "grid" not in st.session_state:
    st.warning("Configure a race first on the **Home Dashboard**.")
    st.stop()

st.caption(f"Race: **{st.session_state['race_label']}**  "
           f"·  {len(st.session_state['grid'])} drivers  "
           f"·  {st.session_state['total_laps']} laps")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    seed = st.number_input("Seed (change for a different race)", value=7, step=1)
    explain = st.checkbox("Narrative (RAG)", value=True)
with col2:
    use_llm = st.checkbox("Refine with Claude Haiku", value=False)
with col3:
    run = st.button("▶️ Simulate", type="primary", use_container_width=True)

if not run and "last_sim" not in st.session_state:
    st.info("Click **Simulate** to run a single seeded race.")
    st.stop()

if run:
    payload = {
        "circuit_id": st.session_state["circuit_id"],
        "total_laps": st.session_state["total_laps"],
        "grid": st.session_state["grid"],
        "strategies": [],
        "weather": st.session_state["weather"],
        "seed": int(seed),
    }
    with st.spinner("Simulating race lap-by-lap…"):
        try:
            st.session_state["last_sim"] = simulate(payload, explain=explain, llm=use_llm)
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

resp = st.session_state["last_sim"]
drivers_map = st.session_state.get("drivers_map", {})

def name(did: str) -> str:
    return drivers_map.get(did, did)

# ---------- Final standings header ----------

m1, m2, m3 = st.columns(3)
m1.metric("🏁 Winner", name(resp["final_standings"][0]["driver_id"]))
m2.metric("Laps run", resp["laps_ran"])
m3.metric("Events logged", len(resp["events"]))

# ---------- Race replay timeline ----------

st.subheader("🎞️ Race replay")

timeline = resp["timeline"]
if not timeline:
    st.warning("No timeline snapshots returned.")
    st.stop()

max_lap = len(timeline)
lap = st.slider("Lap", 1, max_lap, max_lap, 1)
snap = timeline[lap - 1]

lb = pd.DataFrame(snap["standings"])
lb["driver"] = lb["driver_id"].map(name)
lb = lb[["position", "driver", "gap_s", "compound", "tyre_age", "retired"]]
lb.columns = ["P", "driver", "gap (s)", "tyre", "tyre age", "out?"]

st.dataframe(lb, hide_index=True, use_container_width=True)

# ---------- Tyre strategy chart ----------

st.subheader("🟢 Tyre strategy")

rows = []
for i, s in enumerate(timeline, start=1):
    for st_row in s["standings"]:
        rows.append({
            "lap": i,
            "driver": name(st_row["driver_id"]),
            "compound": st_row["compound"],
            "position": st_row["position"],
        })
tyre_df = pd.DataFrame(rows)

# Colour map matching F1 compound conventions.
compound_colours = {"SOFT": "#e83030", "MEDIUM": "#f0c020", "HARD": "#eaeaea"}

fig = px.scatter(
    tyre_df, x="lap", y="driver", color="compound",
    color_discrete_map=compound_colours,
    title="Compound on track per lap",
)
fig.update_traces(marker_size=7)
fig.update_layout(height=max(320, 22 * tyre_df["driver"].nunique()),
                  yaxis_title="", legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

# ---------- Events panel ----------

st.subheader("📣 Events")

events = resp["events"]
if events:
    ev_df = pd.DataFrame(events)
    ev_df["driver"] = ev_df["driver_id"].fillna("—").map(
        lambda d: name(d) if d != "—" else "—"
    )
    ev_df = ev_df[["lap", "kind", "driver", "detail"]]
    kinds = sorted(ev_df["kind"].unique().tolist())
    sel_kinds = st.multiselect("Filter by kind", kinds, default=kinds)
    ev_df = ev_df[ev_df["kind"].isin(sel_kinds)].sort_values("lap")
    st.dataframe(ev_df, hide_index=True, use_container_width=True)
else:
    st.info("No events this race — quiet one.")

# ---------- Narrative ----------

if resp.get("narrative"):
    src = resp.get("narrative_source", "template")
    badge = "🤖 Claude Haiku" if src == "llm" else "📝 template"
    with st.expander(f"🗒️ Narrative ({badge})", expanded=True):
        st.write(resp["narrative"])
