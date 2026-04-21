"""🎮 Race Simulation — lap-by-lap leaderboard, animated replay, race trace."""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import simulate

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
    # Reset replay state for the new simulation.
    st.session_state.pop("replay_lap", None)
    st.session_state.pop("replay_playing", None)

resp = st.session_state["last_sim"]
drivers_map = st.session_state.get("drivers_map", {})

def name(did: str) -> str:
    return drivers_map.get(did, did)

# ---------- Final standings header ----------

m1, m2, m3 = st.columns(3)
m1.metric("🏁 Winner", name(resp["final_standings"][0]["driver_id"]))
m2.metric("Laps run", resp["laps_ran"])
m3.metric("Events logged", len(resp["events"]))

timeline = resp["timeline"]
if not timeline:
    st.warning("No timeline snapshots returned.")
    st.stop()

max_lap = len(timeline)
st.session_state.setdefault("replay_lap", max_lap)
st.session_state.setdefault("replay_playing", False)

# ---------- Race replay ----------

st.subheader("🎞️ Race replay")

play_cols = st.columns([1, 1, 1, 4])
with play_cols[0]:
    if st.button("▶️ Play" if not st.session_state["replay_playing"] else "⏸ Pause",
                 use_container_width=True):
        st.session_state["replay_playing"] = not st.session_state["replay_playing"]
        if st.session_state["replay_playing"] and st.session_state["replay_lap"] >= max_lap:
            st.session_state["replay_lap"] = 1
with play_cols[1]:
    if st.button("⏮ Reset", use_container_width=True):
        st.session_state["replay_lap"] = 1
        st.session_state["replay_playing"] = False
with play_cols[2]:
    speed = st.select_slider("Speed", ["0.5×", "1×", "2×", "4×"], value="2×",
                             label_visibility="collapsed")

lap = st.slider("Lap", 1, max_lap, st.session_state["replay_lap"], 1, key="replay_slider")
# Slider drag wins over any auto-advance.
if lap != st.session_state["replay_lap"]:
    st.session_state["replay_lap"] = lap
    st.session_state["replay_playing"] = False

snap = timeline[st.session_state["replay_lap"] - 1]

lb = pd.DataFrame(snap["standings"])
lb["driver"] = lb["driver_id"].map(name)
lb = lb[["position", "driver", "gap_s", "compound", "tyre_age", "retired"]]
lb.columns = ["P", "driver", "gap (s)", "tyre", "tyre age", "out?"]

st.dataframe(lb, hide_index=True, use_container_width=True, height=min(720, 44 + 36 * len(lb)))

# Incremental auto-play. Streamlit isn't a game loop — we advance one lap per rerun.
if st.session_state["replay_playing"] and st.session_state["replay_lap"] < max_lap:
    delay = {"0.5×": 1.2, "1×": 0.6, "2×": 0.3, "4×": 0.15}[speed]
    time.sleep(delay)
    st.session_state["replay_lap"] += 1
    st.rerun()
elif st.session_state["replay_playing"]:
    st.session_state["replay_playing"] = False  # hit the last lap

# ---------- Animated position chart ----------

st.subheader("🏁 Position animation (Plotly play button)")

rows = []
for i, s in enumerate(timeline, start=1):
    for st_row in s["standings"]:
        rows.append({
            "lap": i,
            "driver_id": st_row["driver_id"],
            "driver": name(st_row["driver_id"]),
            "position": st_row["position"],
            "compound": st_row["compound"],
            "tyre_age": st_row["tyre_age"],
            "gap_s": st_row["gap_s"],
        })
df_all = pd.DataFrame(rows)

compound_colours = {"SOFT": "#e83030", "MEDIUM": "#f0c020", "HARD": "#eaeaea"}
n_drivers = df_all["driver"].nunique()

anim = px.scatter(
    df_all, x="position", y="driver", animation_frame="lap",
    color="compound", color_discrete_map=compound_colours,
    hover_data=["tyre_age", "gap_s"],
    category_orders={"driver": sorted(df_all["driver"].unique())},
    range_x=[0, n_drivers + 1],
)
anim.update_traces(marker_size=14)
anim.update_layout(height=max(360, 28 * n_drivers),
                   yaxis_title="", xaxis_title="track position",
                   legend_title_text="")
st.plotly_chart(anim, use_container_width=True)

# ---------- Race trace ----------

st.subheader("📈 Race trace (position over laps)")

trace = px.line(
    df_all.sort_values(["driver", "lap"]),
    x="lap", y="position", color="driver",
    category_orders={"driver": sorted(df_all["driver"].unique())},
)
trace.update_yaxes(autorange="reversed", dtick=1)
trace.update_traces(line=dict(width=2))
trace.update_layout(height=520, legend=dict(orientation="v", x=1.02, y=1))
st.plotly_chart(trace, use_container_width=True)

# ---------- Tyre strategy gantt ----------

st.subheader("🟢 Tyre strategy (per-driver stints)")

# Build stint segments: runs of identical compound per driver.
stints = []
for drv, sub in df_all.sort_values("lap").groupby("driver"):
    start_lap = None
    prev_comp = None
    for _, row in sub.iterrows():
        if row["compound"] != prev_comp:
            if prev_comp is not None:
                stints.append({"driver": drv, "compound": prev_comp,
                               "start": start_lap, "end": int(row["lap"]) - 1})
            start_lap = int(row["lap"])
            prev_comp = row["compound"]
    stints.append({"driver": drv, "compound": prev_comp,
                   "start": start_lap, "end": int(sub["lap"].max())})

stints_df = pd.DataFrame(stints)
gantt = go.Figure()
for comp, colour in compound_colours.items():
    sub = stints_df[stints_df["compound"] == comp]
    if sub.empty:
        continue
    gantt.add_trace(go.Bar(
        y=sub["driver"], base=sub["start"] - 0.5,
        x=sub["end"] - sub["start"] + 1,
        orientation="h", marker_color=colour, name=comp,
        hovertemplate="%{y}: L%{base:.0f}–L%{x:.0f}<extra>" + comp + "</extra>",
    ))
gantt.update_layout(
    barmode="stack", height=max(320, 24 * n_drivers),
    xaxis_title="lap", yaxis_title="",
    yaxis=dict(categoryorder="array",
               categoryarray=sorted(df_all["driver"].unique(), reverse=True)),
    legend_title_text="",
)
st.plotly_chart(gantt, use_container_width=True)

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
