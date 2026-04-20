"""Shared explanation helper — retrieves RAG context and produces a narrative.

All three endpoints call this with their ML output + a retrieval query. Returns
(narrative_text, source) where source is 'template' or 'llm'. Never raises on
RAG/LLM failures — falls back silently to the deterministic template.
"""

from __future__ import annotations

from typing import Callable, Literal

from rag import index as rag_index
from rag import llm as rag_llm


def explain(
    *,
    retrieval_query: str,
    retrieval_filters: dict | None,
    build_prompt: Callable[[list[dict]], tuple[str, str]],
    use_llm: bool,
    n_context: int = 5,
) -> tuple[str, Literal["template", "llm"]]:
    # Retrieve — gracefully handle a missing/empty index.
    try:
        retrieved = rag_index.query(retrieval_query, n=n_context, **(retrieval_filters or {}))
    except Exception:
        retrieved = []

    template_narrative, llm_user_message = build_prompt(retrieved)

    if not use_llm or not rag_llm.have_key():
        return template_narrative, "template"

    try:
        refined = rag_llm.refine(llm_user_message)
        if refined:
            return refined, "llm"
    except Exception:
        # Never fail the request over an LLM hiccup.
        pass
    return template_narrative, "template"
