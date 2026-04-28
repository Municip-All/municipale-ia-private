#!/usr/bin/env python3
"""
Peuple PostgreSQL (pgvector) avec des signalements synthétiques :
- doublons sémantiques proches,
- spams,
- scores de sentiment variés (basés sur la même heuristique que le Smart-Analyzer).

Usage :
  export DATABASE_URL=postgresql://user:pass@localhost:5432/municipall
  python scripts/seed_synthetic_pg.py [--truncate]
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any

# Permet d'exécuter depuis la racine du dépôt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    from municipal.db import get_connection, get_conninfo
    from municipal.embeddings import embed_texts
    from municipal.spam_sentiment import analyze_spam_sentiment_urgency

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Vide la table reports avant insertion.",
    )
    args = parser.parse_args()

    get_conninfo()

    # Textes de référence (FR) : spams, urgences, proches sémantiquement (doublons)
    items: list[dict[str, Any]] = [
        {
            "content": "Gagnez un iPhone 15 Pro gratuit, cliquez ici : http://spam.example",
            "status": "Spam",
        },
        {
            "content": "Transfert d'argent crypto urgent — félicitations vous avez gagné 5000 EUR",
            "status": "Spam",
        },
        {
            "content": "Trou dans la route rue des Lilas, dangereux pour les vélos",
            "status": "Open",
        },
        {
            "content": "Chaussée déformée rue des Lilas, creux sur la file cyclable",
            "status": "Open",
            "duplicate_of_index": 2,
        },
        {
            "content": "Lampe du lampadaire 12 HS, traversée très sombre la nuit, peur d'agression",
            "status": "Open",
        },
        {
            "content": "Jardin public non tondue depuis des semaines, ronces envahissantes",
            "status": "Open",
        },
        {
            "content": "Mégots et bouteilles près de l'école, déchets partout (merci d'intervenir)",
            "status": "In_Progress",
        },
    ]

    texts = [x["content"] for x in items]
    mat = embed_texts(texts, normalize=True)

    built: list[dict[str, Any]] = []
    id_by_index: list[str] = []
    for i, spec in enumerate(items):
        t = spec["content"]
        meta = analyze_spam_sentiment_urgency(t)
        emb = [float(x) for x in mat[i].tolist()]
        st = spec["status"]
        if st == "Spam":
            rcat, rser = "Autre", "Modération / SPAM"
        else:
            from municipal.router import smart_route

            r = smart_route(t)
            rcat, rser = r["category"], r["municipal_service"]
        built.append(
            {
                "content": t,
                "status": st,
                "category": rcat,
                "municipal_service": rser,
                "sentiment_score": float(meta["sentiment_score"]),
                "embedding": emb,
                "duplicate_of_index": spec.get("duplicate_of_index"),
            }
        )

    with get_connection() as conn:
        with conn.cursor() as cur:
            if args.truncate:
                cur.execute("DELETE FROM reports")
                conn.commit()
        with conn.cursor() as cur:
            for i, row in enumerate(built):
                uid = str(uuid.uuid4())
                st = row["status"]
                dup = None
                vec = row["embedding"]
                vlit = "[" + ",".join(str(x) for x in vec) + "]"
                cur.execute(
                    """
                    INSERT INTO reports (
                      id, user_id, content, category, status, sentiment_score,
                      embedding, duplicate_of_id, municipal_service
                    ) VALUES (
                      %s::uuid, %s::uuid, %s, %s, %s::report_status, %s, %s::vector, %s, %s
                    ) RETURNING id::text
                    """,
                    (
                        uid,
                        str(uuid.uuid4()),
                        row["content"],
                        row["category"],
                        st,
                        row["sentiment_score"],
                        vlit,
                        dup,
                        row["municipal_service"],
                    ),
                )
                r = cur.fetchone()
                if not r:
                    raise RuntimeError("insert failed")
                id_by_index.append(r[0])
            # second pass for duplicate_of_id
            for i, row in enumerate(built):
                idx = row.get("duplicate_of_index")
                if idx is None:
                    continue
                orig = id_by_index[idx]
                cur.execute(
                    """
                    UPDATE reports
                    SET status = 'Duplicate'::report_status, duplicate_of_id = %s::uuid
                    WHERE id = %s::uuid
                    """,
                    (orig, id_by_index[i]),
                )
        conn.commit()

    print(f"OK — {len(built)} signalements insérés (truncate={args.truncate}).")


if __name__ == "__main__":
    main()
