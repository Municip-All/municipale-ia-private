"""Fixtures partagées : client FastAPI « reporting seul », helpers."""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reporting_routes import router as reporting_router


@pytest.fixture
def reporting_app() -> FastAPI:
    """Évite de charger RandomForest / Redis au démarrage de l’API complète."""
    app = FastAPI()
    app.include_router(reporting_router)
    return app


@pytest.fixture
def reporting_client(reporting_app: FastAPI) -> TestClient:
    return TestClient(reporting_app)


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> None:
    from municipal.rate_limit import limiter

    limiter._storage.reset()


@pytest.fixture
def postgres_enabled() -> bool:
    return bool((os.environ.get("DATABASE_URL") or "").strip())


@pytest.fixture
def random_user_id() -> str:
    return str(uuid.uuid4())


def reset_db_connection_cache() -> None:
    import municipal.db as db

    db._conninfo = None
