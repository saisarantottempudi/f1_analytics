"""FastAPI entry point — /predict, /simulate, /h2h endpoints.

Run with:
    uvicorn backend.main:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.h2h import router as h2h_router
from backend.api.predict import router as predict_router
from backend.api.simulate import router as simulate_router

app = FastAPI(
    title="F1 Analytics AI",
    version="0.5.0",
    description=(
        "Race prediction, full-race simulation, and driver head-to-head "
        "analytics backed by FastF1 + Ergast data and an ensemble of "
        "Monte Carlo + LSTM + tabular-Q ML models."
    ),
)

# Streamlit frontend (stage 7) runs on a different port — permit cross-origin calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(simulate_router)
app.include_router(h2h_router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "stage": "5-backend"}
