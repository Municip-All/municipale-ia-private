from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


def test_api_key_rejects_protected_endpoint_without_key():
    with patch.dict("os.environ", {"API_KEY": "test-secret", "NODE_ENV": "production"}, clear=False):
        from api_fastapi import app
        app.state.limiter = MagicMock()
        with TestClient(app) as client:
            r = client.post(
                "/reporting/chat/citoyen",
                json={"user_id": "u1", "message": "test"},
            )
    assert r.status_code == 401


def test_api_key_allows_with_valid_key():
    with patch.dict("os.environ", {"API_KEY": "test-secret", "NODE_ENV": "production"}, clear=False):
        from api_fastapi import app
        app.state.limiter = MagicMock()
        with TestClient(app) as client:
            with patch("reporting_routes.get_conninfo", return_value="postgresql://mock"):
                with patch("reporting_routes.llm_configured", return_value=False):
                    with patch("reporting_routes.submit_report", return_value={
                        "report_id": "r1", "status": "Open", "category": "Voirie",
                        "municipal_service": "ST", "sentiment_score": -0.1,
                        "is_spam": False, "duplicate_of_id": None, "smart_router": {},
                    }):
                        r = client.post(
                            "/reporting/chat/citoyen",
                            json={"user_id": "u1", "message": "test"},
                            headers={"X-API-Key": "test-secret"},
                        )
    assert r.status_code == 200


def test_request_id_middleware_sets_header():
    with patch.dict("os.environ", {}, clear=False):
        from api_fastapi import app
        app.state.limiter = MagicMock()
        with TestClient(app) as client:
            r = client.get("/health")
    assert r.status_code == 200
    assert "x-request-id" in r.headers


def test_request_id_middleware_preserves_provided_id():
    with patch.dict("os.environ", {}, clear=False):
        from api_fastapi import app
        app.state.limiter = MagicMock()
        with TestClient(app) as client:
            r = client.get("/health", headers={"X-Request-ID": "my-correlation-42"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "my-correlation-42"
