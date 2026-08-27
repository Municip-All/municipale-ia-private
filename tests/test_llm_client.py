"""Tests client LLM (LiteLLM) : mock + intégration optionnelle (clé via env uniquement)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from municipal import llm_client as lc


def test_chat_completion_success_parses_choice():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "  Réponse test  \n"

    with patch.object(lc, "LITELLM_API_KEY", "fake-key"):
        with patch.object(lc, "LITELLM_MODEL", "mistral/mistral-medium-2508"):
            with patch.object(lc, "LITELLM_API_BASE", ""):
                with patch.object(lc, "LITELLM_TIMEOUT_S", 120.0):
                    with patch("municipal.llm_client.litellm.completion", return_value=fake_response):
                        out = lc.chat_completion(
                            [{"role": "user", "content": "Bonjour"}],
                            temperature=0.1,
                        )
    assert out == "Réponse test"


def test_chat_completion_raises_if_no_key():
    with patch.object(lc, "LITELLM_API_KEY", ""):
        with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
            lc.chat_completion([{"role": "user", "content": "x"}])


def test_chat_completion_bad_payload_raises():
    fake_response = MagicMock()
    fake_response.choices = []

    with patch.object(lc, "LITELLM_API_KEY", "x"):
        with patch.object(lc, "LITELLM_MODEL", "mistral/mistral-medium"):
            with patch.object(lc, "LITELLM_API_BASE", ""):
                with patch.object(lc, "LITELLM_TIMEOUT_S", 120.0):
                    with patch("municipal.llm_client.litellm.completion", return_value=fake_response):
                        with pytest.raises(RuntimeError, match="inattendue"):
                            lc.chat_completion([{"role": "user", "content": "x"}])


@pytest.mark.integration
def test_llm_live():
    """
    Appel réel — opt-in explicite : RUN_LIVE_LLM_TEST=1 + LITELLM_API_KEY dans l'environnement.
    """
    import os

    if (os.environ.get("RUN_LIVE_LLM_TEST") or "").strip() != "1":
        pytest.skip("RUN_LIVE_LLM_TEST != 1 (intégration opt-in)")
    key = (os.environ.get("LITELLM_API_KEY") or "").strip()
    if not key:
        pytest.skip("LITELLM_API_KEY non définie (intégration)")

    msg = lc.chat_completion(
        [{"role": "user", "content": "Réponds uniquement: OK"}],
        temperature=0,
    )
    assert isinstance(msg, str)
    assert len(msg) >= 1
