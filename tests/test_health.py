from __future__ import annotations

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_health_returns_model_loaded_false_when_no_predictor():
    with patch.dict("os.environ", {}, clear=False):
        from api_fastapi import app
        app.state.limiter = MagicMock()
        with TestClient(app) as client:
            r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "model_loaded" in body
    assert "redis" in body
    assert "database" in body


def test_health_shows_degraded_when_db_unavailable():
    with patch.dict("os.environ", {}, clear=False):
        from api_fastapi import app
        app.state.limiter = MagicMock()
        with TestClient(app) as client:
            with patch("api_fastapi.get_conninfo", side_effect=RuntimeError("no db")):
                r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database"] == "error"
    assert body["status"] == "degraded"
