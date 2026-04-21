"""FastAPI entry point — /predict, /simulate, /h2h endpoints.

Run with:
    uvicorn backend.main:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.api.h2h import router as h2h_router
from backend.api.meta import router as meta_router
from backend.api.predict import router as predict_router
from backend.api.simulate import router as simulate_router

app = FastAPI(
    title="F1 Analytics AI",
    version="0.8.0",
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
app.include_router(meta_router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    # Hitting the bare host would otherwise return {"detail":"Not Found"}.
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    from rag.index import is_ready as rag_ready
    from rag.llm import have_key
    return {
        "status": "ok",
        "stage": "8-polish",
        "rag_index_ready": str(rag_ready()),
        "llm_key_present": str(have_key()),
    }
