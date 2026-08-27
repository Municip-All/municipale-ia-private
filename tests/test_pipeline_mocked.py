"""Pipeline submit_report : comportements attendus avec mocks (sans Postgres)."""

from __future__ import annotations

from unittest.mock import patch

from municipal.pipeline import submit_report


def _fake_analysis(spam: bool = False) -> dict:
    return {
        "is_spam": spam,
        "spam_reasons": ["x"] if spam else [],
        "sentiment_score": -0.2,
        "urgency": "moyenne",
        "tone": "neutre",
        "embedding": [0.01] * 384,
        "embedding_dim": 384,
        "content_preview": "preview",
    }


def test_pipeline_spam_skips_duplicate_check() -> None:
    with patch("municipal.pipeline.smart_analyzer", return_value=_fake_analysis(spam=True)):
        with patch("municipal.pipeline.smart_route", return_value={"category": "Autre", "municipal_service": "X"}):
            with patch("municipal.pipeline.duplicate_finder") as dup:
                with patch("municipal.pipeline.insert_report", return_value="rid-1") as ins:
                    out = submit_report("00000000-0000-0000-0000-000000000001", "spam text")
    dup.assert_not_called()
    ins.assert_called_once()
    call_kw = ins.call_args[1]
    assert call_kw["status"] == "Spam"


def test_pipeline_duplicate_when_finder_matches() -> None:
    with patch("municipal.pipeline.smart_analyzer", return_value=_fake_analysis(spam=False)):
        with patch("municipal.pipeline.smart_route", return_value={"category": "Voirie", "municipal_service": "ST"}):
            with patch(
                "municipal.pipeline.duplicate_finder",
                return_value={"is_duplicate": True, "match_id": "original-uuid"},
            ):
                with patch("municipal.pipeline.insert_report", return_value="rid-2") as ins:
                    out = submit_report("00000000-0000-0000-0000-000000000002", "trou dans la route")
    assert out["status"] == "Doublon"
    assert out["duplicate_of_id"] == "original-uuid"
    assert ins.call_args[1]["status"] == "Doublon"


def test_pipeline_open_when_no_duplicate() -> None:
    with patch("municipal.pipeline.smart_analyzer", return_value=_fake_analysis(spam=False)):
        with patch("municipal.pipeline.smart_route", return_value={"category": "Voirie", "municipal_service": "ST"}):
            with patch(
                "municipal.pipeline.duplicate_finder",
                return_value={"is_duplicate": False, "best_similarity": 0.3},
            ):
                with patch("municipal.pipeline.insert_report", return_value="rid-3"):
                    out = submit_report("00000000-0000-0000-0000-000000000003", "banc cassé")
    assert out["status"] == "En attente"
    assert out["duplicate_of_id"] is None


def test_pipeline_skip_duplicate_check_flag() -> None:
    with patch("municipal.pipeline.smart_analyzer", return_value=_fake_analysis(spam=False)):
        with patch("municipal.pipeline.smart_route", return_value={"category": "X", "municipal_service": "Y"}):
            with patch("municipal.pipeline.duplicate_finder") as dup:
                with patch("municipal.pipeline.insert_report", return_value="rid-4"):
                    out = submit_report(
                        "00000000-0000-0000-0000-000000000004",
                        "texte",
                        skip_duplicate_check=True,
                    )
    dup.assert_not_called()
    assert out["status"] == "En attente"
