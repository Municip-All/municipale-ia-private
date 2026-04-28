from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional
from uuid import UUID

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


def find_nearest_report_by_embedding(
    embedding: list[float],
    exclude_id: str | None,
    threshold: float,
) -> dict[str, Any]:
    """
    Cosinus (pgvector) : similarité = 1 - distance_cosinus, avec vecteurs normalisés.
    """
    if not embedding:
        return {"found": False, "message": "embedding_vide"}
    # Format attendu par pgvector : cast texte explicite, paramètres bind uniquement
    import psycopg
    from psycopg import sql

    vec_literal = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    excl: Optional[UUID] = None
    if exclude_id:
        try:
            excl = UUID(exclude_id)
        except ValueError:
            excl = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            q = """
                SELECT id::text, status::text,
                       1 - (embedding <=> %s::vector) AS sim
                FROM reports
                WHERE (%s::uuid IS NULL OR id IS DISTINCT FROM %s::uuid)
                  AND status::text NOT IN ('Duplicate', 'Spam')
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """
            cur.execute(
                q,
                (vec_literal, excl, excl, vec_literal),
            )
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
    if not user_id:
        raise ValueError("user_id requis")
    uid = UUID(user_id)
    vec = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    dup = UUID(duplicate_of_id) if duplicate_of_id else None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (
                  user_id, content, category, status, sentiment_score,
                  embedding, duplicate_of_id, municipal_service
                ) VALUES (
                  %s, %s, %s, %s::report_status, %s, %s::vector, %s, %s
                ) RETURNING id::text
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
    """Les signalements les plus « urgents » : sentiment le plus négatif (détresse, colère)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, content, category, sentiment_score::float8,
                       status::text, created_at, municipal_service
                FROM reports
                WHERE created_at >= NOW() - (%s::int * INTERVAL '1 day')
                  AND status::text = 'Open'
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
                SET status = 'Duplicate'::report_status, duplicate_of_id = %s::uuid
                WHERE id = %s::uuid
                """,
                (original_id, report_id),
            )
        conn.commit()
