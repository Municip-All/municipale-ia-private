from __future__ import annotations

import json
from typing import Any

from municipal.analyzer import smart_analyzer
from municipal.config import DUPLICATE_SIMILARITY_THRESHOLD
from municipal.db import insert_report
from municipal.duplicate import duplicate_finder
from municipal.router import smart_route


def submit_report(
    user_id: str, content: str, skip_duplicate_check: bool = False, tenant_id: str = "ia-pipeline"
) -> dict[str, Any]:
    """
    Enchaîne Smart-Analyzer → Smart-Router → Duplicate-Finder, puis insère.
    Toute persistance : requêtes SQL paramétrées dans municipal.db.
    """
    a = smart_analyzer(content, user_id)
    r = smart_route(content)
    status = "Open"
    dup_id: str | None = None
    dup_snapshot: dict[str, Any] | None = None
    if a.get("is_spam"):
        status = "Spam"
        dup_snapshot = {"skipped": True, "reason": "spam_detecte"}
    elif not skip_duplicate_check:
        d = duplicate_finder(
            a["embedding"],
            exclude_report_id=None,
            threshold=DUPLICATE_SIMILARITY_THRESHOLD,
        )
        dup_snapshot = {
            key: d.get(key)
            for key in (
                "found",
                "is_duplicate",
                "match_id",
                "match_status",
                "best_similarity",
                "message",
            )
            if key in d
        }
        if d.get("is_duplicate") and d.get("match_id"):
            status = "Duplicate"
            dup_id = d["match_id"]
    else:
        dup_snapshot = {"skipped": True, "reason": "skip_duplicate_check"}
    new_id = insert_report(
        user_id=user_id,
        content=content,
        category=r["category"],
        status=status,
        sentiment_score=float(a["sentiment_score"]),
        embedding=a["embedding"],
        duplicate_of_id=dup_id,
        municipal_service=r["municipal_service"],
        tenant_id=tenant_id,
    )
    return {
        "report_id": new_id,
        "status": status,
        "category": r["category"],
        "municipal_service": r["municipal_service"],
        "sentiment_score": a["sentiment_score"],
        "is_spam": a["is_spam"],
        "duplicate_of_id": dup_id,
        "smart_router": r,
        "analysis": {
            "spam_reasons": a["spam_reasons"],
            "urgency": a["urgency"],
            "tone": a["tone"],
            "embedding_dim": a["embedding_dim"],
            "content_preview": a.get("content_preview", "")[:400],
        },
        "duplicate_check": dup_snapshot,
        "full_text_length": len((content or "").strip()),
    }


def submit_report_json(user_id: str, content: str, tenant_id: str = "ia-pipeline") -> str:
    return json.dumps(submit_report(user_id, content, tenant_id=tenant_id), ensure_ascii=False)
