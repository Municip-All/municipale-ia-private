from unittest.mock import patch

from municipal.duplicate import duplicate_finder


def test_duplicate_finder_returns_found_false_on_empty_embedding():
    with patch("municipal.duplicate.find_nearest_report_by_embedding", return_value={"found": False, "message": "embedding_vide"}):
        out = duplicate_finder([], exclude_report_id=None, threshold=0.85)
    assert out["found"] is False


def test_duplicate_finder_delegates_with_default_threshold():
    call_args = {}

    def capture(*a, **kw):
        call_args.update(args=a, kwargs=kw)
        return {"found": True, "is_duplicate": False, "match_id": None, "best_similarity": 0.3}

    with patch("municipal.duplicate.find_nearest_report_by_embedding", side_effect=capture):
        out = duplicate_finder([0.1] * 384, exclude_report_id=5, threshold=None)

    assert out["found"] is True
    assert call_args["args"][1] == 5
    assert call_args["args"][2] == 0.85


def test_duplicate_finder_uses_custom_threshold():
    captured = {}

    def capture(*a, **kw):
        captured["threshold"] = a[2]
        return {"found": True, "is_duplicate": True, "match_id": 42, "best_similarity": 0.9}

    with patch("municipal.duplicate.find_nearest_report_by_embedding", side_effect=capture):
        out = duplicate_finder([0.1] * 384, threshold=0.95)

    assert captured["threshold"] == 0.95
    assert out["is_duplicate"] is True
