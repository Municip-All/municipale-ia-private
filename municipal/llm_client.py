from __future__ import annotations

import logging
import time
from typing import Optional

import litellm

from municipal.config import (
    LITELLM_API_BASE,
    LITELLM_API_KEY,
    LITELLM_MODEL,
    LITELLM_TIMEOUT_S,
)

litellm.suppress_debug_info = True

logger = logging.getLogger("municipall.llm")

_MAX_RETRIES = 2
_RETRY_BACKOFF_S = 1.5


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
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_S * (2 ** attempt)
                logger.warning("llm attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, e)
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM failed after {_MAX_RETRIES + 1} attempts: {last_exc}") from last_exc
        try:
            return (
                response.choices[0].message.content
                or ""
            ).strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Réponse LLM inattendue: {response!r}") from e
    raise RuntimeError(f"LLM failed after {_MAX_RETRIES + 1} attempts: {last_exc}") from last_exc
