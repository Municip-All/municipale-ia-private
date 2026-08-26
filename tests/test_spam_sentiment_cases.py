"""Cas d’usage : heuristiques spam / sentiment / urgence (sans modèle d’embedding)."""

from __future__ import annotations

import pytest

from municipal.spam_sentiment import analyze_spam_sentiment_urgency


@pytest.mark.parametrize(
    "text,expect_spam",
    [
        ("Gagnez un iPhone gratuit cliquez ici http://spam.io", True),
        ("Transfert crypto urgent félicitations", True),
        ("", True),
        ("a", True),
        ("Nid de poule dangereux rue Victor Hugo", False),
        (
            "fv salut je veut fair ma pub irojgiortgjiortzjgiortnbviortnb",
            True,
        ),
        ("bcdfghjklmnopqrstvwxyzzzzzxxxxxxbbbbbbbb", True),
    ],
)
def test_spam_detection(text: str, expect_spam: bool) -> None:
    out = analyze_spam_sentiment_urgency(text)
    assert out["is_spam"] is expect_spam


def test_sentiment_negative_on_insult_context() -> None:
    out = analyze_spam_sentiment_urgency(
        "C’est inadmissible, trottoir cassé depuis des mois, honte à la mairie !!!"
    )
    assert out["sentiment_score"] < 0
    assert out["urgency"] in ("moyenne", "haute")


def test_sentiment_positive_on_thanks() -> None:
    out = analyze_spam_sentiment_urgency(
        "Merci pour le passage des services, très satisfaits du nouvel éclairage."
    )
    assert out["sentiment_score"] > 0
