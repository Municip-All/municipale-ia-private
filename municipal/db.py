from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from municipal.config import get_database_url

logger = logging.getLogger("municipall.db")

_conninfo: str | None = None


def get_conninfo() -> str:
    global _conninfo
    if _conninfo is not None:
        return _conninfo
    url = get_database_url() or os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL n'est pas définie. Exemple : "
            "postgresql://user:pass@localhost:5432/municipall"
        )
    _conninfo = url
    return _conninfo


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    import psycopg

    with psycopg.connect(get_conninfo(), autocommit=False) as conn:
        yield conn


def enrich_report(
    report_id: int,
    tenant_id: str,
    content: str,
    category: str,
    municipal_service: str,
    sentiment_score: float,
    embedding: list[float],
    is_spam: bool,
    duplicate_of_id: int | None,
) -> None:
    vec = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reports
                SET
                  category          = %s,
                  ai_category       = %s,
                  municipal_service = %s,
                  sentiment_score   = %s,
                  embedding         = %s::vector,
                  is_spam           = %s,
                  duplicate_of_id   = %s,
                  ai_processed      = TRUE
                WHERE id = %s AND tenant_id = %s
                """,
                (
                    category,
                    category,
                    municipal_service,
                    float(sentiment_score),
                    vec,
                    is_spam,
                    int(duplicate_of_id) if duplicate_of_id else None,
                    int(report_id),
                    tenant_id,
                ),
            )
        conn.commit()


def find_nearest_report_by_embedding(
    embedding: list[float],
    exclude_id: int | None,
    threshold: float,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    if not embedding:
        return {"found": False, "message": "embedding_vide"}
    vec_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    tenant_clause = "AND tenant_id = %s" if tenant_id else ""
    tenant_params: tuple[Any, ...] = (tenant_id,) if tenant_id else ()
    with get_connection() as conn:
        with conn.cursor() as cur:
            if exclude_id is not None:
                q = f"""
                    SELECT id, status,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM reports
                    WHERE id <> %s
                      AND status NOT IN ('Duplicate', 'Spam', 'Doublon', 'Rejeté')
                      {tenant_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """
                cur.execute(q, (vec_literal, exclude_id, *tenant_params, vec_literal))
            else:
                q = f"""
                    SELECT id, status,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM reports
                    WHERE status NOT IN ('Duplicate', 'Spam', 'Doublon', 'Rejeté')
                    {tenant_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """
                cur.execute(q, (vec_literal, *tenant_params, vec_literal))
            row = cur.fetchone()
            if not row:
                return {"found": False, "best_similarity": 0.0, "match_id": None}
            match_id, status, sim = row[0], row[1], float(row[2]) if row[2] is not None else 0.0
            if sim > threshold:
                return {
                    "found": True,
                    "is_duplicate": True,
                    "match_id": match_id,
                    "match_status": status,
                    "best_similarity": sim,
                }
            return {
                "found": True,
                "is_duplicate": False,
                "match_id": None,
                "match_status": None,
                "best_similarity": sim,
            }


def _coerce_int(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        logger.warning("%s non numérique ignoré (colonne INTEGER) : %s", field, value)
        return None


def insert_report(
    user_id: str,
    content: str,
    category: str,
    status: str,
    sentiment_score: float,
    embedding: list[float],
    duplicate_of_id: str | None,
    municipal_service: str | None,
    tenant_id: str = "ia-pipeline",
) -> str:
    uid = _coerce_int(user_id, "user_id")
    vec = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    dup = _coerce_int(duplicate_of_id, "duplicate_of_id")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (
                  tenant_id, user_id, description, category, status, sentiment_score,
                  embedding, duplicate_of_id, municipal_service
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s::vector, %s, %s
                ) RETURNING id
                """,
                (
                    tenant_id,
                    uid,
                    content,
                    category,
                    status,
                    float(sentiment_score),
                    vec,
                    dup,
                    municipal_service,
                ),
            )
            r = cur.fetchone()
            conn.commit()
            if not r:
                raise RuntimeError("insertion_échouée")
            return str(r[0])


_REPORT_ORDER_WHITELIST: dict[str, str] = {
    "created_at_desc": "created_at DESC",
    "created_at_asc": "created_at ASC",
    "updated_at_desc": "updated_at DESC",
    "sentiment_asc": "sentiment_score ASC, created_at DESC",
    "sentiment_desc": "sentiment_score DESC, created_at DESC",
    "id_desc": "id DESC",
    "id_asc": "id ASC",
}

_REPORT_GROUP_WHITELIST: dict[str, str] = {
    "category": "category",
    "status": "status",
    "municipal_service": "municipal_service",
    "ai_category": "ai_category",
}

REPORT_ORDER_VALUES: list[str] = list(_REPORT_ORDER_WHITELIST)
REPORT_GROUP_VALUES: list[str] = list(_REPORT_GROUP_WHITELIST)

_REPORT_STATUS_ALIASES: dict[str, str] = {
    "en attente": "En attente",
    "en cours": "En cours",
    "résolu": "Résolu",
    "resolu": "Résolu",
    "doublon": "Doublon",
    "rejeté": "Rejeté",
    "rejete": "Rejeté",
    "spam": "Spam",
    "pending": "En attente",
    "in_progress": "En cours",
    "resolved": "Résolu",
    "closed": "Résolu",
    "rejected": "Rejeté",
    "duplicate": "Doublon",
}


def _normalize_status_list(status: Any) -> list[str]:
    if status is None:
        return []
    raw = status if isinstance(status, (list, tuple)) else [status]
    out: list[str] = []
    for s in raw:
        if s is None:
            continue
        text = str(s).strip()
        if not text:
            continue
        out.append(_REPORT_STATUS_ALIASES.get(text.lower(), text))
    return out


def query_reports(
    status: str | list[str] | None = None,
    category: str | None = None,
    days: int = 30,
    order_by: str = "created_at_desc",
    limit: int = 20,
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    statuses = _normalize_status_list(status)
    order_sql = _REPORT_ORDER_WHITELIST.get(
        str(order_by or "").strip().lower(),
        _REPORT_ORDER_WHITELIST["created_at_desc"],
    )
    n_days = max(1, min(365, _coerce_int(days, "days") or 30))
    conditions = ["created_at >= NOW() - (%s::int * INTERVAL '1 day')"]
    params: list[Any] = [n_days]
    if tenant_id:
        conditions.append("tenant_id = %s")
        params.append(tenant_id)
    if statuses:
        conditions.append("status = ANY(%s)")
        params.append(statuses)
    cat = str(category or "").strip()
    if cat:
        conditions.append("LOWER(category) = LOWER(%s)")
        params.append(cat)
    n_limit = max(1, min(50, _coerce_int(limit, "limit") or 20))
    sql = (
        "SELECT id, tenant_id, user_id, description, category, status, "
        "sentiment_score, ai_confidence, is_spam, duplicate_of_id, "
        "municipal_service, ai_category, created_at, updated_at "
        f"FROM reports WHERE {' AND '.join(conditions)} "
        f"ORDER BY {order_sql} LIMIT %s"
    )
    params.append(n_limit)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "tenant_id": r[1],
                "user_id": r[2],
                "content": r[3],
                "category": r[4],
                "status": r[5],
                "sentiment_score": r[6],
                "ai_confidence": r[7],
                "is_spam": r[8],
                "duplicate_of_id": r[9],
                "municipal_service": r[10],
                "ai_category": r[11],
                "created_at": r[12].isoformat() if r[12] else None,
                "updated_at": r[13].isoformat() if r[13] else None,
            }
        )
    return out


def count_reports(
    group_by: str = "status", days: int | None = None, tenant_id: str | None = None
) -> list[dict[str, Any]]:
    col = _REPORT_GROUP_WHITELIST.get(
        str(group_by or "").strip().lower(), _REPORT_GROUP_WHITELIST["status"]
    )
    conditions: list[str] = []
    params: list[Any] = []
    if tenant_id:
        conditions.append("tenant_id = %s")
        params.append(tenant_id)
    n_days = _coerce_int(days, "days")
    if n_days is not None:
        conditions.append("created_at >= NOW() - (%s::int * INTERVAL '1 day')")
        params.append(max(1, min(365, n_days)))
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (
        f"SELECT {col} AS group_key, COUNT(*) AS count FROM reports {where} "
        "GROUP BY " + col + " ORDER BY count DESC, group_key ASC"
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "group_key": r[0] if r[0] is not None else "non renseigné",
                "count": int(r[1]),
            }
        )
    return out


def top_urgent_by_sentiment(
    days: int = 7, limit: int = 3, tenant_id: str | None = None
) -> list[dict[str, Any]]:
    tenant_clause = "AND tenant_id = %s\n" if tenant_id else ""
    params: list[Any] = [days, *( [tenant_id] if tenant_id else [] ), int(limit)]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, description, category, sentiment_score,
                       status, created_at, municipal_service
                FROM reports
                WHERE created_at >= NOW() - (%s::int * INTERVAL '1 day')
                {tenant_clause}  AND status IN ('En attente', 'Open')
                ORDER BY sentiment_score ASC, created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "content": r[1],
                "category": r[2],
                "sentiment_score": r[3],
                "status": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "municipal_service": r[6],
            }
        )
    return out
