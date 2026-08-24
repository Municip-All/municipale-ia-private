"""
Client LLM universel via LiteLLM — proxy agnostique (Mistral, OpenAI, Anthropic…).
Définir LITELLM_API_KEY ; ne jamais logger la clé.
"""

from __future__ import annotations

from typing import Optional

import litellm

from municipal.config import (
    LITELLM_API_BASE,
    LITELLM_API_KEY,
    LITELLM_MODEL,
    LITELLM_TIMEOUT_S,
)

litellm.suppress_debug_info = True


def llm_configured() -> bool:
    return bool(LITELLM_API_KEY)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    model: Optional[str] = None,
) -> str:
    if not LITELLM_API_KEY:
        raise RuntimeError("LITELLM_API_KEY n'est pas définie.")
    kwargs: dict = {
        "model": model or LITELLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "timeout": LITELLM_TIMEOUT_S,
        "api_key": LITELLM_API_KEY,
    }
    if LITELLM_API_BASE:
        kwargs["api_base"] = LITELLM_API_BASE
    response = litellm.completion(**kwargs)
    try:
        return (
            response.choices[0].message.content
            or ""
        ).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Réponse LLM inattendue: {response!r}") from e
