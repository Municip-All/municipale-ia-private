"""
Démonstration fonctionnelle (réseau + Postgres + embeddings + LLM via LiteLLM).

Lancer depuis la racine du projet :

  RUN_LIVE_DEMO_TERMINAL_TEST=1 DATABASE_URL='postgresql://…' LITELLM_API_KEY='…' \
    pytest tests/test_live_llm_demo_roundtrip.py::test_live_demo_submit_then_llm_reply -s -v

Ou utiliser le script interactif : python scripts/demo_llm_chat.py
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from municipal.llm_client import chat_completion, llm_configured


def _truthy_live_env() -> bool:
    raw = os.environ.get("RUN_LIVE_DEMO_TERMINAL_TEST") or ""
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _postgres_configured() -> bool:
    return bool((os.environ.get("DATABASE_URL") or "").strip())


@pytest.mark.llm_demo
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.postgres
def test_live_demo_submit_then_llm_reply() -> None:
    pytest.importorskip("sentence_transformers")
    from municipal.pipeline import submit_report

    if not _truthy_live_env():
        pytest.skip(
            "Définir RUN_LIVE_DEMO_TERMINAL_TEST=1 avec DATABASE_URL et LITELLM_API_KEY pour ce test réel."
        )
    if not _postgres_configured():
        pytest.skip("DATABASE_URL requise.")
    if not llm_configured():
        pytest.skip("LITELLM_API_KEY requise.")

    uid = str(uuid.uuid4())
    phrase = (
        "Lampadaire en panne sur la place depuis une semaine, "
        "c'est difficile de rentrer tard le soir."
    )

    outcome = submit_report(uid, phrase)
    assert "report_id" in outcome and outcome["status"] in ("En attente", "Doublon", "Spam")
    assert outcome.get("analysis") and "spam_reasons" in outcome["analysis"]

    msgs = [
        {
            "role": "system",
            "content": (
                "Assistant démo Municip'All — résume brièvement le JSON signalement "
                "(statut, thème, sentiment) en français sans jargon backend."
            ),
        },
        {
            "role": "user",
            "content": "Contexte :\n" + json.dumps(outcome, indent=2, ensure_ascii=False, default=str),
        },
    ]
    reply = chat_completion(msgs, temperature=0.2)

    print("\n<<< LLM :\n", reply, "\n", sep="")

    assert isinstance(reply, str)
    assert len(reply.strip()) > 40
