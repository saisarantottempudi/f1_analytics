"""Thin httpx wrapper around the FastAPI backend.

The Streamlit pages only touch the API through these helpers — no direct
parquet reads, no business logic.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.getenv("F1_API_BASE", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=TIMEOUT)


# ---------- meta / health ----------

@st.cache_data(ttl=30)
def health() -> dict[str, Any] | None:
    try:
        with _client() as c:
            r = c.get("/health")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def meta() -> dict[str, Any] | None:
    try:
        with _client() as c:
            r = c.get("/meta")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        st.error(f"Could not reach API at {API_BASE} — {e}")
        return None


# ---------- endpoints ----------

def predict(payload: dict, *, explain: bool = False, llm: bool = False) -> dict:
    with _client() as c:
        r = c.post("/predict", json=payload,
                   params={"explain": explain, "llm": llm})
        r.raise_for_status()
        return r.json()


def simulate(payload: dict, *, explain: bool = False, llm: bool = False) -> dict:
    with _client() as c:
        r = c.post("/simulate", json=payload,
                   params={"explain": explain, "llm": llm})
        r.raise_for_status()
        return r.json()


def h2h(payload: dict, *, explain: bool = False, llm: bool = False) -> dict:
    with _client() as c:
        r = c.post("/h2h", json=payload,
                   params={"explain": explain, "llm": llm})
        r.raise_for_status()
        return r.json()
