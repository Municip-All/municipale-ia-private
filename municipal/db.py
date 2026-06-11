from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from municipal.config import DATABASE_URL

_conninfo: str | None = None


def get_conninfo() -> str:
    global _conninfo
    if _conninfo is not None:
        return _conninfo
    url = DATABASE_URL or os.environ.get("DATABASE_URL", "")
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
    """
    Enrichit un signalement existant créé par le backend NestJS avec les résultats IA.
    Met à jour : category, ai_category, municipal_service, sentiment_score, embedding,
                  is_spam, duplicate_of_id, ai_processed.
    """
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
) -> dict[str, Any]:
    """
    Cosinus (pgvector) : similarité = 1 - distance_cosinus, avec vecteurs normalisés.
    Filtre les signalements non-spam / non-duplicate, exclut le report portant exclude_id.
    """
    if not embedding:
        return {"found": False, "message": "embedding_vide"}
    vec_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    with get_connection() as conn:
        with conn.cursor() as cur:
            if exclude_id is not None:
                q = """
                    SELECT id, status,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM reports
                    WHERE id <> %s
                      AND status NOT IN ('Duplicate', 'Spam', 'Doublon', 'Rejeté')
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """
                cur.execute(q, (vec_literal, exclude_id, vec_literal))
            else:
                q = """
                    SELECT id, status,
                           1 - (embedding <=> %s::vector) AS sim
                    FROM reports
                    WHERE status NOT IN ('Duplicate', 'Spam', 'Doublon', 'Rejeté')
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """
                cur.execute(q, (vec_literal, vec_literal))
            row = cur.fetchone()
            if not row:
                return {"found": False, "best_similarity": 0.0, "match_id": None}
            match_id, status, sim = row[0], row[1], float(row[2])
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


def insert_report(
    user_id: str,
    content: str,
    category: str,
    status: str,
    sentiment_score: float,
    embedding: list[float],
    duplicate_of_id: str | None,
    municipal_service: str | None,
) -> str:
    """Insère un signalement. Compatible schéma unifié INT (NestJS + IA)."""
    try:
        uid = int(user_id) if user_id else None
    except (ValueError, TypeError):
        uid = None
    vec = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    dup = None
    if duplicate_of_id:
        try:
            dup = int(duplicate_of_id)
        except (ValueError, TypeError):
            dup = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (
                  tenant_id, user_id, description, category, status, sentiment_score,
                  embedding, duplicate_of_id, municipal_service
                ) VALUES (
                  'ia-pipeline', %s, %s, %s, %s, %s, %s::vector, %s, %s
                ) RETURNING id
                """,
                (
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


def top_urgent_by_sentiment(
    days: int = 7, limit: int = 3
) -> list[dict[str, Any]]:
    """Les signalements les plus urgents : sentiment le plus négatif."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, description, category, sentiment_score,
                       status, created_at, municipal_service
                FROM reports
                WHERE created_at >= NOW() - (%s::int * INTERVAL '1 day')
                  AND status = 'Open'
                ORDER BY sentiment_score ASC, created_at DESC
                LIMIT %s
                """,
                (days, int(limit)),
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


def update_report_duplicate(
    report_id: str, original_id: str
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reports
                SET status = 'Duplicate', duplicate_of_id = %s
                WHERE id = %s
                """,
                (int(original_id), int(report_id)),
            )
        conn.commit()
