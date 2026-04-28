"""Tests client Mistral (NumSpot) : mock + intégration optionnelle (clé via env uniquement)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from municipal import mistral_client as mc


def test_chat_completion_success_parses_choice():
    fake_json = {
        "choices": [{"message": {"content": "  Réponse test  \n"}}],
        "id": "mock",
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=fake_json)

    with patch.object(mc, "MISTRAL_API_KEY", "fake-key"):
        with patch.object(mc, "MISTRAL_API_BASE", "https://example.com/v1"):
            with patch.object(mc, "MISTRAL_MODEL", "mistral-medium-2508"):
                with patch("httpx.Client") as Client:
                    inst = MagicMock()
                    inst.__enter__.return_value = inst
                    inst.__exit__.return_value = None
                    inst.post.return_value = mock_resp
                    Client.return_value = inst
                    out = mc.chat_completion(
                        [{"role": "user", "content": "Bonjour"}],
                        temperature=0.1,
                    )
    assert out == "Réponse test"


def test_chat_completion_raises_if_no_key():
    with patch.object(mc, "MISTRAL_API_KEY", ""):
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            mc.chat_completion([{"role": "user", "content": "x"}])


def test_chat_completion_bad_payload_raises():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={})

    with patch.object(mc, "MISTRAL_API_KEY", "x"):
        with patch.object(mc, "MISTRAL_API_BASE", "https://example.com/v1"):
            with patch.object(mc, "MISTRAL_MODEL", "mistral-medium"):
                with patch("httpx.Client") as Client:
                    inst = MagicMock()
                    inst.__enter__.return_value = inst
                    inst.__exit__.return_value = None
                    inst.post.return_value = mock_resp
                    Client.return_value = inst
                    with pytest.raises(RuntimeError, match="inattendue"):
                        mc.chat_completion([{"role": "user", "content": "x"}])


@pytest.mark.integration
def test_mistral_live_numspot():
    """
    Appel réel — exiger MISTRAL_API_KEY dans l'environnement (ne jamais commiter la clé).
    Échoue avec message explicite si 401 (auth / clé).
    """
    import os

    key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
    if not key:
        pytest.skip("MISTRAL_API_KEY non définie (intégration)")

    try:
        msg = mc.chat_completion(
            [{"role": "user", "content": "Réponds uniquement: OK"}],
            temperature=0,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            pytest.fail(
                "401 Unauthorized : le serveur refuse le jeton Bearer. "
                "Vérifier que la clé est valide, non expirée, et le bon périmètre NumSpot ; "
                "suivre la doc fournisseur pour obtenir un token d’accès si nécessaire."
            )
        raise
    assert isinstance(msg, str)
    assert len(msg) >= 1
