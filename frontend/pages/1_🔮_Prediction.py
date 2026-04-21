"""🔮 Future Prediction Panel — Monte Carlo-backed race probabilities."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import predict

st.set_page_config(page_title="Prediction · F1 Analytics AI", page_icon="🔮", layout="wide")
st.title("🔮 Race Prediction")

if "grid" not in st.session_state:
    st.warning("Configure a race first on the **Home Dashboard**.")
    st.stop()

st.caption(f"Race: **{st.session_state['race_label']}**  "
           f"·  {len(st.session_state['grid'])} drivers  "
           f"·  rain {st.session_state['weather']['rain_probability']:.0%}")

col1, col2 = st.columns([3, 1])
with col1:
    explain = st.checkbox("Narrative explanation (RAG template)", value=True)
    use_llm = st.checkbox("Refine with Claude Haiku (requires API key)", value=False)
with col2:
    run = st.button("▶️ Run prediction", type="primary", use_container_width=True)

if not run:
    st.info("Click **Run prediction** to run Monte Carlo over the selected grid.")
    st.stop()

payload = {
    "circuit_id": st.session_state["circuit_id"],
    "total_laps": st.session_state["total_laps"],
    "grid": st.session_state["grid"],
    "weather": st.session_state["weather"],
    "n_sims": st.session_state["n_sims"],
}

with st.spinner(f"Running {payload['n_sims']} Monte Carlo sims…"):
    try:
        resp = predict(payload, explain=explain, llm=use_llm)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

# ---------- Headline stats ----------

drivers_map = st.session_state.get("drivers_map", {})

def name(did: str) -> str:
    return drivers_map.get(did, did)

m1, m2, m3 = st.columns(3)
m1.metric("🏆 Predicted winner", name(resp["winner_driver_id"]))
m2.metric("📍 Pole", name(resp["pole_driver_id"]))
m3.metric("🚨 Safety-car probability", f"{resp['sc_probability']:.0%}")

# ---------- Probability table + chart ----------

st.subheader("Per-driver probabilities")

df = pd.DataFrame(resp["drivers"])
df["driver"] = df["driver_id"].map(name)
df = df.sort_values("p_win", ascending=False).reset_index(drop=True)

bar = px.bar(
    df.head(12),
    x="driver", y=["p_win", "p_podium", "p_points"],
    barmode="group",
    labels={"value": "probability", "variable": "outcome"},
    title="Win / Podium / Points probability",
)
bar.update_layout(xaxis_tickangle=-30, height=420,
                  yaxis_tickformat=".0%", legend_title_text="")
st.plotly_chart(bar, use_container_width=True)

st.dataframe(
    df[["grid_position", "driver", "mean_finish", "p_win", "p_podium", "p_points"]]
      .rename(columns={"grid_position": "grid",
                       "mean_finish": "mean finish",
                       "p_win": "P(win)",
                       "p_podium": "P(podium)",
                       "p_points": "P(points)"}),
    use_container_width=True,
    hide_index=True,
    column_config={
        "P(win)":    st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
        "P(podium)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
        "P(points)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
    },
)

# ---------- Narrative ----------

if resp.get("narrative"):
    src = resp.get("narrative_source", "template")
    badge = "🤖 Claude Haiku" if src == "llm" else "📝 template"
    with st.expander(f"🗒️ Narrative ({badge})", expanded=True):
        st.write(resp["narrative"])
