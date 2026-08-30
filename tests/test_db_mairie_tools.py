"""Tests unitaires query_reports / count_reports : SQL paramétré, whitelists, bornes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any
from unittest.mock import patch

from municipal.db import count_reports, query_reports

_ROWS: list[tuple[Any, ...]] = [
    (
        12,
        "tenant-1",
        7,
        "Lampadaire HS rue des Écoles",
        "Éclairage public",
        "En cours",
        -0.4,
        0.8,
        False,
        None,
        "Services techniques",
        "Éclairage public",
        datetime(2026, 8, 30, 9, 0, 0),
        datetime(2026, 8, 30, 9, 5, 0),
    )
]


class _FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.calls.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class _FakeConn:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


def _run(db_func, rows: list[tuple[Any, ...]], *args: Any, **kwargs: Any):
    fake = _FakeConn(rows)
    with patch("municipal.db.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = fake
        result = db_func(*args, **kwargs)
    sql, params = fake.cursor_obj.calls[0]
    return result, sql, params


class TestQueryReportsSql:
    def test_filters_are_parameterized(self) -> None:
        result, sql, params = _run(
            query_reports,
            _ROWS,
            status="En cours",
            category="Voirie",
            days=45,
            order_by="sentiment_asc",
            limit=5,
        )
        assert "En cours" not in sql
        assert "Voirie" not in sql
        assert "status = ANY(%s)" in sql
        assert "LOWER(category) = LOWER(%s)" in sql
        assert "ORDER BY sentiment_score ASC, created_at DESC" in sql
        assert "LIMIT %s" in sql
        assert params == (45, ["En cours"], "Voirie", 5)
        assert result[0]["content"] == "Lampadaire HS rue des Écoles"
        assert result[0]["created_at"] == "2026-08-30T09:00:00"

    def test_unknown_order_by_falls_back_to_whitelist_default(self) -> None:
        _, sql, params = _run(query_reports, _ROWS, order_by="id; DROP TABLE reports")
        assert "ORDER BY created_at DESC" in sql
        assert "DROP" not in sql.upper().replace("ORDER BY", "")
        assert params == (30, 20)

    def test_status_aliases_normalized(self) -> None:
        _, sql, params = _run(query_reports, _ROWS, status="en cours")
        assert "En cours" not in sql
        assert params == (30, ["En cours"], 20)
        _, _, params = _run(query_reports, _ROWS, status="resolved")
        assert params == (30, ["Résolu"], 20)
        _, _, params = _run(query_reports, _ROWS, status=["doublon", "spam"])
        assert params == (30, ["Doublon", "Spam"], 20)

    def test_days_and_limit_clamped(self) -> None:
        _, _, params = _run(query_reports, _ROWS, days=9999, limit=9999)
        assert params == (365, 50)
        _, _, params = _run(query_reports, _ROWS, days=0, limit=0)
        assert params == (30, 20)

    def test_no_filters_still_parameterized(self) -> None:
        _, sql, params = _run(query_reports, _ROWS)
        assert "status = ANY" not in sql
        assert "LOWER(category)" not in sql
        assert params == (30, 20)


class TestCountReportsSql:
    def test_group_by_whitelisted(self) -> None:
        _, sql, params = _run(count_reports, [("Voirie", 12)], group_by="category", days=14)
        assert "GROUP BY category" in sql
        assert "ORDER BY count DESC, group_key ASC" in sql
        assert params == (14,)
        assert params[0] == 14

    def test_unknown_group_by_falls_back_to_status(self) -> None:
        _, sql, params = _run(count_reports, [("En attente", 4)], group_by="tenant_id; DELETE")
        assert "GROUP BY status" in sql
        assert "DELETE" not in sql.upper()
        assert params == ()

    def test_optional_days_parameterized(self) -> None:
        _, _, params = _run(count_reports, [("Voirie", 12)], group_by="category")
        assert params == ()
        _, _, params = _run(count_reports, [("Voirie", 12)], group_by="category", days=9999)
        assert params == (365,)

    def test_rows_mapped_with_null_group(self) -> None:
        result, _, _ = _run(count_reports, [(None, 3)], group_by="municipal_service")
        assert result == [{"group_key": "non renseigné", "count": 3}]
