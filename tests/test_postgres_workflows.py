"""Workflows bout-en-bout avec une vraie base PostgreSQL + pgvector."""

from __future__ import annotations

import pytest

from municipal.db import get_conninfo
from municipal.pipeline import submit_report
from tests.conftest import reset_db_connection_cache
from tests.db_helpers import delete_reports_by_ids


@pytest.fixture
def require_postgres(postgres_enabled: bool) -> None:
    if not postgres_enabled:
        pytest.skip("DATABASE_URL absente : tests postgres ignorés")
    reset_db_connection_cache()
    get_conninfo()


@pytest.mark.postgres
def test_submit_spam_persisted(require_postgres: None, random_user_id: str) -> None:
    r = submit_report(
        random_user_id,
        "Gagnez un iPhone gratuit cliquez ici bit.ly/spam",
    )
    assert r["status"] == "Spam"
    assert r["is_spam"] is True
    rid = r["report_id"]
    assert rid
    delete_reports_by_ids([rid])


@pytest.mark.postgres
def test_submit_open_then_duplicate_same_text(require_postgres: None, random_user_id: str) -> None:
    """Deux envois identiques : le second doit être classé Duplicate et pointe vers le premier."""
    text = "CI-DUPLICATE-TEST lampe cassée même emplacement rue XYZ-123456"
    first = submit_report(random_user_id, text)
    second = submit_report(random_user_id, text)
    try:
        assert first["status"] == "En attente"
        assert second["status"] == "Doublon"
        assert second["duplicate_of_id"] == first["report_id"]
    finally:
        delete_reports_by_ids([second["report_id"], first["report_id"]])


@pytest.mark.postgres
def test_top_urgent_lists_most_negative_sentiment(require_postgres: None, random_user_id: str) -> None:
    from municipal.db import top_urgent_by_sentiment

    a = submit_report(
        random_user_id,
        "SOS aidez-moi effondrement trottoir danger immédiat pour les enfants !!!",
        skip_duplicate_check=True,
    )
    b = submit_report(
        random_user_id,
        "Le bac à fleurs du square est un peu fané",
        skip_duplicate_check=True,
    )
    try:
        assert a["status"] == "En attente" and b["status"] == "En attente"
        rows = top_urgent_by_sentiment(days=7, limit=100)
        ids = [r["id"] for r in rows]
        assert a["report_id"] in ids and b["report_id"] in ids
        assert ids.index(a["report_id"]) < ids.index(b["report_id"])
    finally:
        delete_reports_by_ids([a["report_id"], b["report_id"]])
