"""
Client Mistral via API compatible OpenAI (NumSpot) — POST {base}/chat/completions.
Définir MISTRAL_API_KEY ; ne jamais logger la clé.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from municipal.config import (
    MISTRAL_API_BASE,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    MISTRAL_TIMEOUT_S,
)


def mistral_configured() -> bool:
    return bool(MISTRAL_API_KEY)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    model: Optional[str] = None,
) -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY n'est pas définie.")
    m = model or MISTRAL_MODEL
    url = f"{MISTRAL_API_BASE}/chat/completions"
    payload: dict[str, Any] = {
        "model": m,
        "messages": messages,
        "temperature": temperature,
    }
    with httpx.Client(timeout=MISTRAL_TIMEOUT_S) as client:
        r = client.post(
            url,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
        )
    r.raise_for_status()
    data = r.json()
    try:
        return (
            data["choices"][0]["message"]["content"]
            or ""
        ).strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Réponse API Mistral inattendue: {data!r}") from e
