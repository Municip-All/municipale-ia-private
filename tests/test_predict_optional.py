"""Route /predict : uniquement si les artefacts Random Forest sont présents."""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api_fastapi import app


def _artifacts_ready() -> bool:
    base = Path(__file__).resolve().parent.parent / "artifacts"
    return all(
        (base / name).exists()
        for name in ("tfidf.joblib", "model_rf.joblib", "geo_onehot.joblib")
    )


@pytest.mark.skipif(not _artifacts_ready(), reason="Artefacts ML absents (lancer train_model.py)")
def test_predict_health_and_classification() -> None:
    import api_fastapi
    api_fastapi._API_KEY = ""
    api_fastapi._IS_PROD = False
    api_fastapi.app.state.limiter = MagicMock()
    with TestClient(api_fastapi.app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json().get("model_loaded") is True
        r = client.post(
            "/predict",
            json={
                "description": "tas de déchets",
                "lat": 49.26,
                "lon": 2.44,
                "hour": 10,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert "pred" in body and "proba" in body
    assert 0.0 <= float(body["proba"]) <= 1.0
