"""Utilitaires SQL réservés aux tests (nettoyage)."""

from __future__ import annotations

from municipal.db import get_connection


def delete_reports_by_ids(report_ids: list[str | int]) -> None:
    if not report_ids:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rid in report_ids:
                cur.execute("DELETE FROM reports WHERE id = %s", (int(rid),))
        conn.commit()
