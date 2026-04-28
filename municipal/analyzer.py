from __future__ import annotations

import json
from typing import Any

from municipal.embeddings import embed_one
from municipal.spam_sentiment import analyze_spam_sentiment_urgency


def smart_analyzer(content: str, user_id: str | None = None) -> dict[str, Any]:
    """
    Smart-Analyzer : spam, sentiment/urgence/ton, embedding (local sentence-transformers).
    """
    meta = analyze_spam_sentiment_urgency(content)
    emb = embed_one(content)
    out: dict[str, Any] = {
        "user_id": user_id,
        "content_preview": (content or "")[:400],
        "is_spam": meta["is_spam"],
        "spam_reasons": meta["spam_reasons"],
        "sentiment_score": meta["sentiment_score"],
        "urgency": meta["urgency"],
        "tone": meta["tone"],
        "embedding": emb,
        "embedding_dim": len(emb),
    }
    return out


def smart_analyzer_json(content: str, user_id: str | None = None) -> str:
    return json.dumps(smart_analyzer(content, user_id), ensure_ascii=False)
