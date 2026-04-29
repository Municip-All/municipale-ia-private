"""API /reporting : cas d’usage avec mocks (pas de Postgres ni Mistral)."""

from __future__ import annotations

from unittest.mock import patch
from fastapi.testclient import TestClient

def _report_open() -> dict:
    return {
        "report_id": "r1",
        "status": "Open",
        "category": "Voirie",
        "municipal_service": "Services techniques",
        "sentiment_score": -0.1,
        "is_spam": False,
        "duplicate_of_id": None,
        "smart_router": {},
    }


def _report_spam() -> dict:
    d = _report_open()
    d["status"] = "Spam"
    d["is_spam"] = True
    d["category"] = "Autre"
    return d


class TestReportingSubmitMocked:
    def test_submit_returns_200_with_mocked_db(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.submit_report", return_value=_report_open()):
                r = reporting_client.post(
                    "/reporting/submit",
                    json={"user_id": "00000000-0000-0000-0000-000000000099", "content": "Test"},
                )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "Open"
        assert body["category"] == "Voirie"


class TestCitoyenChatMocked:
    def test_chat_open_message_template_if_no_mistral(
        self, reporting_client: TestClient
    ) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.submit_report", return_value=_report_open()):
                with patch("reporting_routes.mistral_configured", return_value=False):
                    r = reporting_client.post(
                        "/reporting/chat/citoyen",
                        json={
                            "user_id": "00000000-0000-0000-0000-000000000088",
                            "message": "Trou dans la route",
                        },
                    )
        assert r.status_code == 200
        assert "Voirie" in r.json()["reply"]

    def test_chat_spam_tone(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.submit_report", return_value=_report_spam()):
                with patch("reporting_routes.mistral_configured", return_value=False):
                    r = reporting_client.post(
                        "/reporting/chat/citoyen",
                        json={
                            "user_id": "00000000-0000-0000-0000-000000000077",
                            "message": "spam",
                        },
                    )
        assert r.status_code == 200
        assert r.json()["reassured"] is False
        assert "modération" in r.json()["reply"].lower()


class TestMairieChatMocked:
    def test_mairie_keyword_fallback_without_mistral(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.mistral_configured", return_value=False):
                r = reporting_client.post(
                    "/reporting/chat/mairie",
                    json={"query": "bonjour"},
                )
        assert r.status_code == 200
        assert "démo" in r.json()["answer"].lower() or "exemple" in r.json()["answer"].lower()

    def test_mairie_top3_list_without_mistral(self, reporting_client: TestClient) -> None:
        rows = [
            {
                "id": "1",
                "content": "Urgence éclairage",
                "category": "Éclairage public",
                "sentiment_score": -0.9,
                "status": "Open",
                "created_at": None,
                "municipal_service": "ST",
            }
        ]
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.top_urgent_by_sentiment", return_value=rows):
                with patch("reporting_routes.mistral_configured", return_value=False):
                    r = reporting_client.post(
                        "/reporting/chat/mairie",
                        json={"query": "Quels sont les 3 problèmes les plus urgents cette semaine ?"},
                    )
        assert r.status_code == 200
        data = r.json()
        assert len(data["top_reports"]) == 1
        assert "Éclairage" in data["answer"] or "Urgence" in data["answer"]

    def test_mairie_with_mistral_mocked(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.top_urgent_by_sentiment", return_value=[]):
                with patch("reporting_routes.mistral_configured", return_value=True):
                    with patch("reporting_routes.chat_completion", return_value="Réponse synthétique."):
                        r = reporting_client.post(
                            "/reporting/chat/mairie",
                            json={"query": "Résumé des urgences ?"},
                        )
        assert r.status_code == 200
        assert r.json()["answer"] == "Réponse synthétique."
