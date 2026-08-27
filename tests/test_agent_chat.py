"""Tests /reporting/chat/agent : boucle d'agent LLM mockée, outils, fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

QUESTION_URGENT = "Quels sont les 3 problèmes les plus urgents cette semaine ?"


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _urgent_rows() -> list[dict]:
    return [
        {
            "id": 1,
            "content": "Nid de poule dangereux",
            "category": "Voirie",
            "sentiment_score": -0.9,
            "status": "En attente",
            "created_at": "2026-08-27T10:00:00",
            "municipal_service": "Services techniques",
        }
    ]


class TestAgentChatToolFlow:
    def test_top_urgent_tool_then_final_answer(self, reporting_client: TestClient) -> None:
        rows = _urgent_rows()
        llm_responses = [
            _message(tool_calls=[_tool_call("top_urgent_by_sentiment", {"days": 7, "limit": 3})]),
            _message(content="Voici les 3 problèmes les plus urgents cette semaine."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.agent_chat.chat_completion_tools", side_effect=fake_tools) as mock_llm:
                    with patch("municipal.agent_chat.top_urgent_by_sentiment", return_value=rows) as mock_top:
                        r = reporting_client.post(
                            "/reporting/chat/agent", json={"question": QUESTION_URGENT}
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Voici les 3 problèmes les plus urgents cette semaine."
        assert data["top_reports"] == rows
        assert data["tools_used"] == ["top_urgent_by_sentiment"]
        assert data["fallback"] is False
        mock_top.assert_called_once_with(days=7, limit=3)
        assert mock_llm.call_count == 2
        second_messages = mock_llm.call_args_list[1][0][0]
        tool_messages = [m for m in second_messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0]["content"])[0]["category"] == "Voirie"

    def test_analyzer_route_tools_stripped_embedding(self, reporting_client: TestClient) -> None:
        analyzer_result = {
            "is_spam": False,
            "sentiment_score": -0.6,
            "urgency": "high",
            "embedding": [0.1] * 384,
        }
        llm_responses = [
            _message(
                tool_calls=[
                    _tool_call("smart_analyzer", {"text": "Éclairage cassé"}, "call_a"),
                    _tool_call("smart_route", {"text": "Éclairage cassé"}, "call_b"),
                ]
            ),
            _message(content="Analyse : sentiment négatif, catégorie Éclairage public."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.agent_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch("municipal.agent_chat.smart_analyzer", return_value=analyzer_result):
                        with patch(
                            "municipal.agent_chat.smart_route",
                            return_value={"category": "Éclairage public", "municipal_service": "Services techniques", "confidence": 0.7},
                        ):
                            r = reporting_client.post(
                                "/reporting/chat/agent", json={"question": "Analyse ce signalement : Éclairage cassé"}
                            )
        assert r.status_code == 200
        data = r.json()
        assert data["tools_used"] == ["smart_analyzer", "smart_route"]
        assert len(data["analyses"]) == 2
        assert "embedding" not in data["analyses"][0]["result"]
        assert data["analyses"][1]["result"]["category"] == "Éclairage public"
        assert data["top_reports"] == []

    def test_tool_error_is_surfaced_in_trace(self, reporting_client: TestClient) -> None:
        llm_responses = [
            _message(tool_calls=[_tool_call("outil_inexistant", {})]),
            _message(content="Je ne peux pas récupérer ces données."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.agent_chat.chat_completion_tools", side_effect=fake_tools):
                    r = reporting_client.post(
                        "/reporting/chat/agent", json={"question": "Données ?"}
                    )
        assert r.status_code == 200
        data = r.json()
        assert data["analyses"][0]["error"].startswith("outil_inconnu")

    def test_max_four_iterations_then_forced_summary(self, reporting_client: TestClient) -> None:
        looping = _message(tool_calls=[_tool_call("smart_route", {"text": "x"})])
        calls: list[list | None] = []

        def fake_tools(messages, tools=None, **kwargs):
            calls.append(tools)
            if tools:
                return _message(tool_calls=[_tool_call("smart_route", {"text": "x"})])
            return _message(content="Synthèse finale après outillage.")

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.agent_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch("municipal.agent_chat.smart_route", return_value={"category": "Voirie"}):
                        r = reporting_client.post(
                            "/reporting/chat/agent", json={"question": "Itère"}
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Synthèse finale après outillage."
        assert data["fallback"] is False
        assert len(data["tools_used"]) == 4
        assert len(calls) == 5
        assert calls[-1] is None

    def test_tool_arguments_bounded(self, reporting_client: TestClient) -> None:
        rows = _urgent_rows()
        llm_responses = [
            _message(
                tool_calls=[_tool_call("top_urgent_by_sentiment", {"days": 5000, "limit": 999})]
            ),
            _message(content="Réponse."),
        ]

        def fake_tools(messages, tools=None, **kwargs):
            return llm_responses.pop(0)

        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch("municipal.agent_chat.chat_completion_tools", side_effect=fake_tools):
                    with patch("municipal.agent_chat.top_urgent_by_sentiment", return_value=rows) as mock_top:
                        reporting_client.post("/reporting/chat/agent", json={"question": QUESTION_URGENT})
        mock_top.assert_called_once_with(days=90, limit=20)


class TestAgentChatFallback:
    def test_fallback_urgent_without_llm(self, reporting_client: TestClient) -> None:
        rows = _urgent_rows()
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=False):
                with patch("municipal.agent_chat.top_urgent_by_sentiment", return_value=rows):
                    with patch("municipal.agent_chat.chat_completion_tools") as mock_llm:
                        r = reporting_client.post(
                            "/reporting/chat/agent", json={"question": QUESTION_URGENT}
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["fallback"] is True
        assert data["top_reports"] == rows
        assert "urgents" in data["answer"]
        mock_llm.assert_not_called()

    def test_fallback_when_llm_raises(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=True):
                with patch(
                    "municipal.agent_chat.chat_completion_tools",
                    side_effect=RuntimeError("LLM down"),
                ):
                    with patch("municipal.agent_chat.top_urgent_by_sentiment", return_value=[]):
                        r = reporting_client.post(
                            "/reporting/chat/agent", json={"question": QUESTION_URGENT}
                        )
        assert r.status_code == 200
        data = r.json()
        assert data["fallback"] is True
        assert data["top_reports"] == []
        assert "Aucun signalement ouvert" in data["answer"]

    def test_fallback_generic_question_without_llm(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            with patch("reporting_routes.llm_configured", return_value=False):
                r = reporting_client.post(
                    "/reporting/chat/agent", json={"question": "bonjour"}
                )
        assert r.status_code == 200
        data = r.json()
        assert data["fallback"] is True
        assert data["top_reports"] == []

    def test_question_too_long_rejected(self, reporting_client: TestClient) -> None:
        with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
            r = reporting_client.post(
                "/reporting/chat/agent", json={"question": "a" * 2001}
            )
        assert r.status_code == 422

    def test_db_unreachable_returns_503(self, reporting_client: TestClient) -> None:
        with patch(
            "reporting_routes.get_conninfo",
            side_effect=RuntimeError("DATABASE_URL n'est pas définie."),
        ):
            r = reporting_client.post("/reporting/chat/agent", json={"question": "test"})
        assert r.status_code == 503
