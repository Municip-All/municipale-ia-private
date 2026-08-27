from unittest.mock import patch

import pytest

from municipal.config import _build_database_url


def test_build_database_url_uses_env_url():
    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://u:p@db:5432/testdb"}, clear=False):
        import municipal.config as cfg
        cfg._database_url = None
        try:
            assert _build_database_url() == "postgresql://u:p@db:5432/testdb"
        finally:
            cfg._database_url = None


def test_build_database_url_trust_auth_no_password():
    with patch.dict("os.environ", {"DATABASE_URL": "", "DATABASE_PASSWORD": "", "DATABASE_USER": "mehmet", "DATABASE_HOST": "localhost", "DATABASE_PORT": "5432", "DATABASE_NAME": "municipall"}, clear=False):
        import municipal.config as cfg
        cfg._database_url = None
        try:
            url = _build_database_url()
            assert url == "postgresql://mehmet@localhost:5432/municipall"
        finally:
            cfg._database_url = None


def test_build_database_url_constructs_from_parts():
    with patch.dict(
        "os.environ",
        {
            "DATABASE_URL": "",
            "DATABASE_HOST": "myhost",
            "DATABASE_PORT": "5433",
            "DATABASE_USER": "admin",
            "DATABASE_PASSWORD": "secret",
            "DATABASE_NAME": "mydb",
        },
        clear=False,
    ):
        import municipal.config as cfg
        cfg._database_url = None
        try:
            url = _build_database_url()
            assert url == "postgresql://admin:secret@myhost:5433/mydb"
        finally:
            cfg._database_url = None
