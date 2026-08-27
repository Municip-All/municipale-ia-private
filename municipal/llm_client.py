from __future__ import annotations

import logging
import time
from typing import Any, Optional

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


def _completion_with_retries(**kwargs: Any) -> Any:
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
            return response.choices[0].message
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Réponse LLM inattendue: {response!r}") from e
    raise RuntimeError(f"LLM failed after {_MAX_RETRIES + 1} attempts: {last_exc}")


def _base_kwargs(model: Optional[str], temperature: float) -> dict[str, Any]:
    if not LITELLM_API_KEY:
        raise RuntimeError("LITELLM_API_KEY n'est pas définie.")
    kwargs: dict[str, Any] = {
        "model": model or LITELLM_MODEL,
        "temperature": temperature,
        "timeout": LITELLM_TIMEOUT_S,
        "api_key": LITELLM_API_KEY,
    }
    if LITELLM_API_BASE:
        kwargs["api_base"] = LITELLM_API_BASE
    return kwargs


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    model: Optional[str] = None,
) -> str:
    kwargs = _base_kwargs(model, temperature)
    kwargs["messages"] = messages
    message = _completion_with_retries(**kwargs)
    return (getattr(message, "content", None) or "").strip()


def chat_completion_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float = 0.3,
    model: Optional[str] = None,
) -> Any:
    kwargs = _base_kwargs(model, temperature)
    kwargs["messages"] = messages
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return _completion_with_retries(**kwargs)
