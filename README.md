# 🏎️ F1 Analytics AI

An end-to-end Formula 1 analytics and simulation platform that predicts race outcomes, simulates full races lap-by-lap, and compares drivers head-to-head — backed by real F1 data (FastF1 + Ergast), machine learning (LSTM + RL + Monte Carlo), and a retrieval-augmented prompt layer.

> **Status:** 🚧 Step 4 of 8 — ML core (LSTM + Monte Carlo + RL) shipped.

---

## ✨ What it does

| Mode | Input | Output |
|---|---|---|
| 🔮 **Race Prediction** | Circuit + drivers + weather | Winner / pole probabilities, SC/VSC likelihood |
| 🎮 **Race Simulation** | Circuit + grid + weather | Lap-by-lap standings, tyre strategies, pit timing, incidents |
| ⚔️ **Head-to-Head** | Two drivers | Quali / race / consistency / circuit-specific edge |

---

## 🏗️ Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│ FastF1 API   │     │ Ergast API    │     │ Weather (real    │
│ (telemetry)  │     │ (results      │     │  or FastF1 hist) │
│              │     │  since 1950)  │     │                  │
└──────┬───────┘     └───────┬───────┘     └────────┬─────────┘
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Ingestion + Parquet  │  data/raw → data/processed
                  │ cache (scripts/)     │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Feature engineering  │  notebooks/ + backend/features
                  └──────────┬───────────┘
                             ▼
      ┌──────────────────────┼──────────────────────┐
      ▼                      ▼                      ▼
┌───────────┐         ┌──────────────┐       ┌──────────────┐
│ LSTM      │         │ Monte Carlo  │       │ RL agent     │
│ predictor │         │ simulator    │       │ (pit strat)  │
└─────┬─────┘         └──────┬───────┘       └──────┬───────┘
      └──────────────────────┼──────────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ FastAPI backend      │  /predict  /simulate  /h2h
                  │  + RAG (Chroma)      │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Streamlit frontend   │  4 screens (Dashboard /
                  │                      │  Sim / H2H / Prediction)
                  └──────────────────────┘
```

---

## 🧰 Tech stack

- **Data:** FastF1, Ergast API, (optional) OpenWeatherMap
- **ML:** PyTorch (LSTM), scikit-learn, XGBoost, custom RL agent, Monte Carlo
- **RAG:** ChromaDB + sentence-transformers
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit + Plotly
- **Hardware target:** Apple M5, 16 GB unified memory

---

## 🚀 Quick start

```bash
# 1. Clone + enter
git clone https://github.com/saisarantottempudi/f1-analytics-ai.git
cd f1-analytics-ai

# 2. Create venv + install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ingest data
python scripts/ingest_ergast.py --from 2000 --to 2025      # ~1.5 min, 500 races
python scripts/ingest_fastf1.py --from 2022 --to 2025      # ~20–30 min, downloads cache
python scripts/verify_data.py                              # prints row counts

# 4. Build engineered features (Stage 3)
python scripts/build_features.py                           # → data/processed/features.parquet
python notebooks/01_eda.py                                 # regenerates charts in notebooks/figures/

# 5. Train the ML layer (Stage 4)
python scripts/train_predictor.py --epochs 60              # LSTM → models/lstm_predictor.pt
python scripts/run_simulation.py --sims 1000               # Monte Carlo demo
python scripts/train_rl_pit.py --episodes 1500             # RL pit-strategy → models/rl_pit_q.json

# 4. (step 7+) Launch UI
streamlit run frontend/streamlit_app.py
```

---

## 📁 Project layout

```
f1_project/
├── backend/         # FastAPI app, feature engineering, model wrappers
├── frontend/        # Streamlit app + pages
├── data/
│   ├── raw/         # Parquet dumps from FastF1 / Ergast (gitignored)
│   └── processed/   # Engineered features (gitignored)
├── models/          # Trained model weights (gitignored)
├── notebooks/       # EDA + training notebooks
├── rag/             # Chroma index + prompt templates
├── scripts/         # Ingestion + training CLIs
└── tests/
```

---

## 📈 Project status — 8-stage roadmap

Each stage ends with a git commit + push. The README is refreshed at each stage.

- [x] **1. Scaffold + Git init** — repo skeleton, README, `.gitignore`, deps pinned
- [x] **2. Data layer** — Ergast/Jolpica + FastF1 ingestion → Parquet cache; weather client
- [x] **3. Feature engineering + EDA** — 15 features across 6 families; EDA notebook with 5 charts
- [x] **4. ML core** — LSTM (PyTorch + MPS), Monte Carlo simulator, tabular-Q RL pit agent
- [ ] **5. FastAPI backend** — `/predict`, `/simulate`, `/h2h` endpoints
- [ ] **6. RAG layer** — Chroma vector DB over race summaries + prompt templates
- [ ] **7. Streamlit frontend** — Dashboard / Simulation / H2H / Prediction screens
- [ ] **8. Replay animation + polish** — lap-by-lap animation, docs, demo GIF

---

## 📊 Stage 3 — features shipped

The feature table ([backend/features/build.py](backend/features/build.py), produced by [scripts/build_features.py](scripts/build_features.py)) holds one row per (driver × race) with **6 feature families, computed strictly causally** (race N never sees data from race ≥ N):

| Family | Columns |
|---|---|
| Driver rolling form | `drv_rollN_avg_finish_5`, `drv_rollN_dnf_rate_5`, `drv_rollN_points_avg_5` |
| Driver × circuit history | `drv_circ_races_prior`, `drv_circ_avg_finish_prior`, `drv_circ_best_prior` |
| Consistency | `drv_season_finish_std`, `drv_season_finish_iqr` |
| Constructor form | `team_rollN_avg_finish_3` |
| Teammate H2H | `tmate_quali_win_rate_season`, `tmate_race_win_rate_season` |
| Circuit priors | `circ_pole_to_win_rate`, `circ_dnf_rate`, `circ_grid_finish_corr` |

EDA charts — rendered from the 2024 slice, will stabilise once 2000-2025 ingest completes:

| | |
|---|---|
| ![grid vs finish](notebooks/figures/02_grid_vs_finish.png) | ![form vs actual](notebooks/figures/03_form_vs_actual.png) |
| ![teammate h2h](notebooks/figures/04_teammate_h2h.png) | ![pole to win](notebooks/figures/05_pole_to_win.png) |

Form→finish Pearson ρ ≈ **0.62** on the 2024 slice — the rolling feature alone carries substantial signal before any ML.

---

## 🧠 ML core (stage 4 — shipped)

| Component | Lives in | What it does |
|---|---|---|
| **LSTM predictor** | [backend/ml/predictor.py](backend/ml/predictor.py) | Stacked LSTM + categorical embeddings (driver / team / circuit) over a 5-race window. MC-dropout at inference yields a per-driver finish-position distribution. Trains on MPS in seconds. |
| **Monte Carlo simulator** | [backend/ml/monte_carlo.py](backend/ml/monte_carlo.py) | Lap-by-lap race sim with compound-specific tyre deg, fuel burn, pit loss, SC/VSC triggers, and DNF sampling. 1 000 sims in ~5 s. |
| **Tabular Q pit agent** | [backend/ml/rl_pit.py](backend/ml/rl_pit.py) | Q-learning over a discretised (phase, tyre-age, pace-rank, laps-left) state. Learns when to stay out vs. pit for Soft/Medium/Hard. |

Why tabular Q (not DQN): the state space fits in a dict and trains in seconds on a laptop. Upgrade path is DQN if the discretisation hurts us later.

Trained artifacts live in `models/` — `.pt` binaries are gitignored (re-train after clone), the small Q-table JSON ships in-repo so demos work out-of-the-box.

### 📦 Hybrid RAG layer (stage 6)

Retrieves relevant historical race summaries so the explanation layer grounds its narrative in real past incidents.

---

## 📜 License

TBD — add before public release.

---

*Built by [@saisarantottempudi](https://github.com/saisarantottempudi).*
