# 🏎️ F1 Analytics AI

An end-to-end Formula 1 analytics and simulation platform that predicts race outcomes, simulates full races lap-by-lap, and compares drivers head-to-head — backed by real F1 data (FastF1 + Ergast), machine learning (LSTM + RL + Monte Carlo), and a retrieval-augmented prompt layer.

> **Status:** 🚧 Step 1 of 8 — scaffolding in progress.

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

# 3. (step 2+) Ingest data
python scripts/ingest_ergast.py --since 2018
python scripts/ingest_fastf1.py --season 2024

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
- [ ] **2. Data layer** — FastF1 + Ergast ingestion → Parquet cache; weather client
- [ ] **3. Feature engineering + EDA** — driver/team/circuit/consistency features; notebook
- [ ] **4. ML core** — LSTM race predictor, Monte Carlo simulator, RL pit-strategy agent
- [ ] **5. FastAPI backend** — `/predict`, `/simulate`, `/h2h` endpoints
- [ ] **6. RAG layer** — Chroma vector DB over race summaries + prompt templates
- [ ] **7. Streamlit frontend** — Dashboard / Simulation / H2H / Prediction screens
- [ ] **8. Replay animation + polish** — lap-by-lap animation, docs, demo GIF

---

## 🧠 ML approach (stages 3–4)

- **LSTM time-series model** over per-lap features (tyre age, gap ahead, weather) → predicts finishing-position distribution.
- **Monte Carlo simulator** runs 1 000 race replays per scenario; aggregates win / podium / points probabilities and SC/VSC likelihood.
- **RL agent (tabular Q-learning first, upgrade to DQN if needed)** learns optimal pit-lap given tyre life, gap, and race phase.
- **Hybrid RAG layer** retrieves relevant historical race summaries so the explanation layer grounds its narrative in real past incidents.

---

## 📜 License

TBD — add before public release.

---

*Built by [@saisarantottempudi](https://github.com/saisarantottempudi).*
