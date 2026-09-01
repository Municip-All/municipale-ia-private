"""Tests /reporting/chat/citoyen : boucle d'agent citoyen, question vs signalement, fallback pipeline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

EVENTS_ROWS = [
    {
        "titre": "Fête de la musique",
        "description": "Concerts gratuits",
        "lieu": "Place du Marché",
        "date_debut": "2026-06-21T18:00:00",
        "date_fin": "2026-06-22T00:00:00",
        "categorie": "Culture",
    }
]

REPORT_OPEN = {
    "report_id": "12",
    "status": "En attente",
    "category": "Éclairage public",
    "municipal_service": "Services techniques",
    "sentiment_score": -0.6,
    "is_spam": False,
    "duplicate_of_id": None,
}

REPORT_SPAM = {
    "report_id": "13",
    "status": "Spam",
    "category": "Autre",
    "municipal_service": "Modération",
    "sentiment_score": 0.1,
    "is_spam": True,
    "duplicate_of_id": None,
}


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class TestCitoyenQuestionMode:
    def test_question_uses_data_tools_and_never_creates_report(
        self, reporting_client: TestClient
    ) -> None:
        llm_responses = [
            _message(tool_calls=[_tool_call("get_city_events", {"limit": 5})]),
            _message(content="Oui : la Fête de la musique aura lieu le 21 juin place du Marché."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.citoyen_chat.get_city_events", return_value=EVENTS_ROWS
                    ) as mock_events:
                        with patch("reporting_routes.submit_report") as mock_submit:
                            with patch("municipal.citoyen_chat.submit_report") as mock_submit_tool:
                                r = reporting_client.post(
                                    "/reporting/chat/citoyen",
                                    json={
                                        "user_id": "00000000-0000-0000-0000-000000000011",
                                        "message": "Y a-t-il des événements ce mois-ci ?",
                                    },
                                )
        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Oui : la Fête de la musique aura lieu le 21 juin place du Marché."
        assert data["category"] == ""
        assert data["municipal_service"] == ""
        assert data["sentiment_score"] == 0.0
        assert data["reassured"] is True
        mock_events.assert_called_once_with("ia-pipeline", limit=5)
        mock_submit.assert_not_called()
        mock_submit_tool.assert_not_called()

    def test_transport_tool_result_fed_back_to_llm(self, reporting_client: TestClient) -> None:
        llm_responses = [
            _message(tool_calls=[_tool_call("get_transport_disruptions", {"lat": 48.85, "lon": 2.35})]),
            _message(content="Aucune perturbation signalée actuellement."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.citoyen_chat.get_transport_disruptions",
                        return_value={"disruptions": [], "note": "Aucune perturbation signalée."},
                    ) as mock_tr:
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "u1", "message": "Bus perturbés ?"},
                        )
        assert r.status_code == 200
        assert "perturbation" in r.json()["reply"]
        mock_tr.assert_called_once_with("ia-pipeline", lat=48.85, lon=2.35)

    def test_tool_error_surfaced_then_final_answer(self, reporting_client: TestClient) -> None:
        llm_responses = [
            _message(tool_calls=[_tool_call("get_construction_works", {})]),
            _message(content="Je ne peux pas consulter les travaux pour le moment."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.citoyen_chat.get_construction_works",
                        side_effect=RuntimeError("db down"),
                    ):
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "u1", "message": "Travaux rue de la République ?"},
                        )
        assert r.status_code == 200
        assert r.json()["reply"] == "Je ne peux pas consulter les travaux pour le moment."


class TestCitoyenSignalementMode:
    def test_signalement_calls_create_signalement_tool(
        self, reporting_client: TestClient
    ) -> None:
        llm_responses = [
            _message(
                tool_calls=[
                    _tool_call("create_signalement", {"texte": "Éclairage cassé rue des Lilas"})
                ]
            ),
            _message(content="Votre signalement a bien été enregistré."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.citoyen_chat.submit_report", return_value=dict(REPORT_OPEN)
                    ) as mock_submit:
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={
                                "user_id": "user-9",
                                "message": "Éclairage cassé rue des Lilas",
                            },
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Votre signalement a bien été enregistré."
        assert data["category"] == "Éclairage public"
        assert data["municipal_service"] == "Services techniques"
        assert data["sentiment_score"] == -0.6
        assert data["reassured"] is True
        mock_submit.assert_called_once_with("user-9", "Éclairage cassé rue des Lilas", tenant_id="ia-pipeline")

    def test_spam_report_reassured_false(self, reporting_client: TestClient) -> None:
        llm_responses = [
            _message(tool_calls=[_tool_call("create_signalement", {"texte": "spam"})]),
            _message(content="Votre message a été enregistré pour modération."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.citoyen_chat.submit_report", return_value=dict(REPORT_SPAM)
                    ):
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "user-9", "message": "spam"},
                        )
        assert r.status_code == 200
        assert r.json()["reassured"] is False

    def test_report_kept_when_llm_dies_after_creation(self, reporting_client: TestClient) -> None:
        def fake_tools(messages, tools=None, **kwargs):
            if tools:
                return _message(
                    tool_calls=[_tool_call("create_signalement", {"texte": "Nid de poule"})]
                )
            raise RuntimeError("LLM down")

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.citoyen_chat.submit_report", return_value=dict(REPORT_OPEN)
                    ):
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "user-9", "message": "Nid de poule"},
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["category"] == "Éclairage public"
        assert "enregistré" in data["reply"]


class TestCitoyenFallback:
    def test_no_llm_keeps_current_pipeline_template_behavior(
        self, reporting_client: TestClient
    ) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=False):
                with patch("reporting_routes.submit_report", return_value=REPORT_OPEN) as mock_submit:
                    with patch("municipal.citoyen_chat.chat_completion_tools") as mock_llm:
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "u1", "message": "Trou dans la route"},
                        )
        assert r.status_code == 200
        data = r.json()
        assert "Éclairage public" in data["reply"]
        assert data["category"] == "Éclairage public"
        assert data["reassured"] is True
        mock_submit.assert_called_once_with("u1", "Trou dans la route", tenant_id="ia-pipeline")
        mock_llm.assert_not_called()

    def test_llm_error_falls_back_to_pipeline_template(
        self, reporting_client: TestClient
    ) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch(
                    "municipal.citoyen_chat.chat_completion_tools",
                    side_effect=RuntimeError("LLM down"),
                ):
                    with patch(
                        "reporting_routes.chat_completion",
                        side_effect=RuntimeError("LLM down"),
                    ):
                        with patch("reporting_routes.submit_report", return_value=REPORT_OPEN):
                            r = reporting_client.post(
                                "/reporting/chat/citoyen",
                                json={"user_id": "u1", "message": "Trou dans la route"},
                            )
        assert r.status_code == 200
        data = r.json()
        assert "Éclairage public" in data["reply"]
        assert data["category"] == "Éclairage public"

    def test_message_too_long_rejected(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            r = reporting_client.post(
                "/reporting/chat/citoyen",
                json={"user_id": "u1", "message": "a" * 5001},
            )
        assert r.status_code == 422

    def test_db_unreachable_returns_503(self, reporting_client: TestClient) -> None:
        with patch(
            "reporting_routes.get_conninfo",
            side_effect=RuntimeError("DATABASE_URL n'est pas définie."),
        ):
            r = reporting_client.post("/reporting/chat/citoyen", json={"user_id": "u1", "message": "test"})
        assert r.status_code == 503


class TestCitoyenAgentLoopBounds:
    def test_max_four_iterations_then_forced_summary(self, reporting_client: TestClient) -> None:
        calls: list[list | None] = []

        def fake_tools(messages, tools=None, **kwargs):
            calls.append(tools)
            if tools:
                return _message(tool_calls=[_tool_call("smart_route", {"text": "x"})])
            return _message(content="Synthèse finale.")

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch(
                        "municipal.agent_chat.smart_route",
                        return_value={"category": "Voirie"},
                    ):
                        r = reporting_client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "u1", "message": "Itère"},
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["reply"] == "Synthèse finale."
        assert data["category"] == ""
        assert len(calls) == 5
        assert calls[-1] is None

    def test_unknown_tool_error_does_not_crash(self, reporting_client: TestClient) -> None:
        llm_responses = [
            _message(tool_calls=[_tool_call("outil_inexistant", {})]),
            _message(content="Je ne peux pas récupérer ces données."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.citoyen_chat.chat_completion_tools", side_effect=fake_tools):
                    r = reporting_client.post(
                        "/reporting/chat/citoyen",
                        json={"user_id": "u1", "message": "Données ?"},
                    )
        assert r.status_code == 200
        assert r.json()["reply"] == "Je ne peux pas récupérer ces données."

    def test_empty_llm_reply_falls_back_to_pipeline(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch(
                    "municipal.citoyen_chat.chat_completion_tools",
                    return_value=_message(content=""),
                ):
                    with patch(
                        "reporting_routes.chat_completion",
                        side_effect=RuntimeError("LLM down"),
                    ):
                        with patch("reporting_routes.submit_report", return_value=REPORT_OPEN):
                            r = reporting_client.post(
                                "/reporting/chat/citoyen",
                                json={"user_id": "u1", "message": "Trou dans la route"},
                            )
        assert r.status_code == 200
        assert "Éclairage public" in r.json()["reply"]
