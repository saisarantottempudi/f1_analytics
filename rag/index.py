"""Chroma persistent index over the race-narrative corpus.

Uses Chroma's built-in ONNX MiniLM embedder — no separate model download,
runs fully offline after first use. Index lives under rag/chroma_db/.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.corpus_builder import RaceDoc

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / "rag" / "chroma_db"
COLLECTION = "f1_races"


def _client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def build_index(docs: list[RaceDoc], reset: bool = True) -> int:
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"description": "Per-race narrative summaries (Ergast 2000+)"},
    )
    if not docs:
        return 0
    ids = [d.doc_id for d in docs]
    texts = [d.text for d in docs]
    metas = [d.metadata for d in docs]
    # Batch to keep memory small — Chroma's default embedder loads ONNX lazily.
    BATCH = 200
    for i in range(0, len(docs), BATCH):
        coll.add(
            ids=ids[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metas[i:i+BATCH],
        )
    return coll.count()


def query(
    text: str,
    *,
    n: int = 5,
    circuit_id: str | None = None,
    season_from: int | None = None,
    season_to: int | None = None,
) -> list[dict]:
    """Return top-`n` race docs similar to `text`, optionally filtered."""
    client = _client()
    try:
        coll = client.get_collection(COLLECTION)
    except Exception:
        return []

    where_clauses: list[dict] = []
    if circuit_id:
        where_clauses.append({"circuit_id": circuit_id})
    if season_from is not None:
        where_clauses.append({"season": {"$gte": season_from}})
    if season_to is not None:
        where_clauses.append({"season": {"$lte": season_to}})
    where: dict | None
    if not where_clauses:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    res = coll.query(query_texts=[text], n_results=n, where=where)
    out: list[dict] = []
    for i, doc_id in enumerate(res["ids"][0]):
        out.append({
            "id": doc_id,
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i] if res.get("distances") else None,
        })
    return out


def is_ready() -> bool:
    """Cheap check — used by API layer to short-circuit if no index exists."""
    try:
        coll = _client().get_collection(COLLECTION)
        return coll.count() > 0
    except Exception:
        return False
