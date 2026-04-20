"""Anthropic client wrapper with graceful no-key fallback + prompt caching.

Env vars (the client auto-reads ANTHROPIC_API_KEY; CLAUDE_API_KEY is also
honoured as a convenience alias):
    ANTHROPIC_API_KEY   primary
    CLAUDE_API_KEY      alias — if set, copied into ANTHROPIC_API_KEY at import

Model defaults to claude-haiku-4-5-20251001 (cheap + fast) because the user
has limited credits. Opus/Sonnet can be chosen via F1_LLM_MODEL.

Prompt caching is enabled on the system prompt so repeated explanations
only pay for the (small) user-message delta.
"""

from __future__ import annotations

import os

from rag.prompts import SYSTEM_EXPLAINER

DEFAULT_MODEL = os.getenv("F1_LLM_MODEL", "claude-haiku-4-5-20251001")
_MAX_TOKENS = 400

# Env alias — some users set CLAUDE_API_KEY; the SDK expects ANTHROPIC_API_KEY.
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("CLAUDE_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


def have_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def refine(user_message: str, *, model: str = DEFAULT_MODEL) -> str:
    """Call Claude to refine a templated narrative. Raises if no key is set."""
    if not have_key():
        raise RuntimeError(
            "No ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in env — set it before "
            "calling with llm=true, or use the default template narrative."
        )
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_EXPLAINER,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()
