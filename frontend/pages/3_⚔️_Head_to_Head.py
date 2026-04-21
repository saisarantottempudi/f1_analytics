"""⚔️ Head-to-Head — two-driver comparison across quali / race / consistency."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api_client import h2h, meta

st.set_page_config(page_title="Head-to-Head · F1 Analytics AI",
                   page_icon="⚔️", layout="wide")
st.title("⚔️ Driver Head-to-Head")

m = meta()
if not m:
    st.stop()

drivers = m["drivers"]
driver_ids = [d["driver_id"] for d in drivers]
driver_labels = {d["driver_id"]: f"{d['driver_name']} ({d['constructor_id']})"
                 for d in drivers}

def name(did: str) -> str:
    return driver_labels.get(did, did)

col1, col2, col3 = st.columns(3)
with col1:
    da = st.selectbox("Driver A", driver_ids, index=0,
                      format_func=lambda d: driver_labels[d])
with col2:
    default_b = 1 if len(driver_ids) > 1 else 0
    db = st.selectbox("Driver B", driver_ids, index=default_b,
                      format_func=lambda d: driver_labels[d])
with col3:
    circuit_filter = st.selectbox("Circuit (optional)",
                                  ["(all)"] + m["circuits"], index=0)

seasons = m.get("seasons", [])
if seasons:
    s_lo, s_hi = int(min(seasons)), int(max(seasons))
    season_range = st.slider("Season range", s_lo, s_hi, (s_lo, s_hi))
else:
    season_range = None

colx, coly = st.columns([3, 1])
with colx:
    explain = st.checkbox("Narrative (RAG)", value=True)
    use_llm = st.checkbox("Refine with Claude Haiku", value=False)
with coly:
    run = st.button("⚔️ Compare", type="primary", use_container_width=True)

if da == db:
    st.error("Pick two different drivers.")
    st.stop()

if not run:
    st.info("Click **Compare** to run the analysis.")
    st.stop()

payload: dict = {"driver_a": da, "driver_b": db}
if circuit_filter != "(all)":
    payload["circuit_id"] = circuit_filter
if season_range:
    payload["season_from"], payload["season_to"] = season_range

try:
    resp = h2h(payload, explain=explain, llm=use_llm)
except Exception as e:
    st.error(f"H2H failed: {e}")
    st.stop()

# ---------- Headline ----------

winner_label = {"A": name(da), "B": name(db), "TIE": "Tie"}[resp["overall_winner"]]
m1, m2, m3 = st.columns(3)
m1.metric("🏆 Overall winner", winner_label)
m2.metric("Edge", f"{resp['overall_edge_pct']:.1f}%")
m3.metric("Shared races", resp["shared_races"])

# ---------- Sections breakdown ----------

st.subheader("Breakdown")

sections = resp["sections"]
sec_df = pd.DataFrame(sections)
sec_df["A"] = sec_df["a_value"]
sec_df["B"] = sec_df["b_value"]
sec_df["winner"] = sec_df["winner"].map(
    {"A": f"← {name(da)}", "B": f"→ {name(db)}", "TIE": "tie"}
).fillna("—")

st.dataframe(
    sec_df[["label", "A", "B", "winner"]],
    hide_index=True, use_container_width=True,
)

# ---------- Radar ----------

st.subheader("Profile (lower = better on all axes)")

radar_labels = [s["label"] for s in sections
                if s["a_value"] is not None and s["b_value"] is not None][:6]

def _vals(side: str) -> list[float]:
    return [float(s[f"{side}_value"]) for s in sections
            if s["a_value"] is not None and s["b_value"] is not None][:6]

if radar_labels:
    a_vals, b_vals = _vals("a"), _vals("b")
    max_vals = [max(a, b) or 1.0 for a, b in zip(a_vals, b_vals)]
    a_norm = [a / m for a, m in zip(a_vals, max_vals)]
    b_norm = [b / m for b, m in zip(b_vals, max_vals)]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=a_norm + [a_norm[0]],
                                    theta=radar_labels + [radar_labels[0]],
                                    fill="toself", name=name(da)))
    radar.add_trace(go.Scatterpolar(r=b_norm + [b_norm[0]],
                                    theta=radar_labels + [radar_labels[0]],
                                    fill="toself", name=name(db)))
    radar.update_layout(height=500, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(radar, use_container_width=True)

# ---------- Narrative ----------

if resp.get("narrative"):
    src = resp.get("narrative_source", "template")
    badge = "🤖 Claude Haiku" if src == "llm" else "📝 template"
    with st.expander(f"🗒️ Narrative ({badge})", expanded=True):
        st.write(resp["narrative"])
