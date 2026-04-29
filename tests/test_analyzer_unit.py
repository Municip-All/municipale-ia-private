"""Analyzer : structure du résultat avec embedding mocké (pas de sentence-transformers)."""

from __future__ import annotations

from unittest.mock import patch

from municipal.analyzer import smart_analyzer


def test_smart_analyzer_returns_embedding_and_spam_flags() -> None:
    fake = [0.01] * 384
    with patch("municipal.analyzer.embed_one", return_value=fake):
        out = smart_analyzer("Lampadaire cassé place de la République", "user-uuid-placeholder")
    assert out["embedding_dim"] == 384
    assert out["embedding"] == fake
    assert out["is_spam"] is False
    assert -1.0 <= out["sentiment_score"] <= 1.0


def test_smart_analyzer_marks_spam_without_needing_embedding_shape() -> None:
    with patch("municipal.analyzer.embed_one", return_value=[0.0] * 384):
        out = smart_analyzer("Gagnez 5000 EUR crypto gratuitement", None)
    assert out["is_spam"] is True
    assert out["spam_reasons"]
