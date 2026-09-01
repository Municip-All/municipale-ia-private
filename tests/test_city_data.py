"""Tests municipal/city_data.py : SQL paramétré, mapping, fallback HTTP."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import municipal.city_data as city_data


def _fake_conn(rows: list[tuple], capture: dict) -> MagicMock:
    cur = MagicMock()
    conn = MagicMock()

    def execute(query, params=None):
        capture["query"] = query
        capture["params"] = params
        cur.fetchall.return_value = rows

    cur.execute.side_effect = execute
    conn.cursor.return_value.__enter__.return_value = cur
    return conn


class TestGetConstructionWorks:
    def test_select_parametrized_and_mapping(self) -> None:
        rows = [
            (
                "Réfection de la chaussée",
                "Rabotage et enrobé neuf",
                "Rue de la République",
                datetime(2026, 9, 1, 8, 0),
                datetime(2026, 9, 30, 18, 0),
                "En cours",
                "Circulation alternée",
            )
        ]
        capture: dict = {}
        conn = _fake_conn(rows, capture)
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_construction_works("tenant-a", limit=5)
        assert '"tenantId" = %s' in capture["query"]
        assert "LIMIT %s" in capture["query"]
        assert "tenant-a" not in capture["query"]
        assert capture["params"] == ("tenant-a", 5)
        assert out == [
            {
                "titre": "Réfection de la chaussée",
                "description": "Rabotage et enrobé neuf",
                "lieu": "Rue de la République",
                "date_debut": "2026-09-01T08:00:00",
                "date_fin": "2026-09-30T18:00:00",
                "statut": "En cours",
                "impact": "Circulation alternée",
            }
        ]

    def test_limit_clamped(self) -> None:
        capture: dict = {}
        conn = _fake_conn([], capture)
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            city_data.get_construction_works("tenant-a", limit=999)
        assert capture["params"] == ("tenant-a", 50)

    def test_invalid_limit_falls_back_to_default(self) -> None:
        assert city_data._clamp_limit("abc") == 10
        assert city_data._clamp_limit(None) == 10


class TestGetCityEvents:
    def test_select_parametrized_and_mapping(self) -> None:
        rows = [
            (
                "Fête de la musique",
                "Concerts gratuits",
                "Place du Marché",
                datetime(2026, 6, 21, 18, 0),
                datetime(2026, 6, 22, 0, 0),
                "Culture",
            )
        ]
        capture: dict = {}
        conn = _fake_conn(rows, capture)
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_city_events("tenant-a", limit=3)
        assert "WHERE city_id = %s" in capture["query"]
        assert "LIMIT %s" in capture["query"]
        assert "tenant-a" not in capture["query"]
        assert capture["params"] == ("tenant-a", 3)
        assert out == [
            {
                "titre": "Fête de la musique",
                "description": "Concerts gratuits",
                "lieu": "Place du Marché",
                "date_debut": "2026-06-21T18:00:00",
                "date_fin": "2026-06-22T00:00:00",
                "categorie": "Culture",
            }
        ]


class TestGetTransportDisruptions:
    def test_success_extracts_disrupted_lines(self) -> None:
        payload = {
            "lines": [
                {
                    "lineId": "L1",
                    "lineName": "Ligne 1",
                    "mode": "tram",
                    "status": "disrupted",
                    "messages": ["Trafic interrompu entre A et B"],
                },
                {"lineId": "L2", "lineName": "Ligne 2", "mode": "bus", "status": "normal", "messages": []},
            ]
        }
        captured_url: dict = {}
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")

        def fake_urlopen(request, timeout=None):
            captured_url["url"] = getattr(request, "full_url", str(request))
            captured_url["timeout"] = timeout
            ctx = MagicMock()
            ctx.__enter__.return_value = resp
            return ctx

        with patch.dict("os.environ", {"BACKEND_URL": "http://backend:4000"}):
            with patch.object(city_data.urllib.request, "urlopen", side_effect=fake_urlopen):
                out = city_data.get_transport_disruptions("city-1", lat=48.85, lon=2.35, timeout_s=2)
        assert captured_url["url"] == "http://backend:4000/municipalities/city-1/transports/disruptions?lat=48.85&lon=2.35"
        assert captured_url["timeout"] == 2
        assert out == {
            "disruptions": [
                {
                    "ligne": "Ligne 1",
                    "mode": "tram",
                    "messages": ["Trafic interrompu entre A et B"],
                }
            ],
            "note": None,
        }

    def test_backend_down_graceful_fallback(self) -> None:
        with patch.dict("os.environ", {"BACKEND_URL": "http://backend:4000"}):
            with patch.object(
                city_data.urllib.request, "urlopen", side_effect=OSError("connection refused")
            ):
                out = city_data.get_transport_disruptions("city-1", lat=48.85, lon=2.35)
        assert out["disruptions"] == []
        assert out["note"]

    def test_missing_coordinates_no_http_call(self) -> None:
        with patch.object(city_data.urllib.request, "urlopen") as mock_urlopen:
            out = city_data.get_transport_disruptions("city-1")
        assert out["disruptions"] == []
        assert out["note"]
        mock_urlopen.assert_not_called()

    def test_invalid_coordinates_no_http_call(self) -> None:
        with patch.object(city_data.urllib.request, "urlopen") as mock_urlopen:
            out = city_data.get_transport_disruptions("city-1", lat=200, lon=2.35)
        assert out["disruptions"] == []
        assert out["note"]
        mock_urlopen.assert_not_called()

    def test_city_id_url_encoded(self) -> None:
        captured_url: dict = {}
        resp = MagicMock()
        resp.read.return_value = json.dumps({"lines": []}).encode("utf-8")

        def fake_urlopen(request, timeout=None):
            captured_url["url"] = getattr(request, "full_url", str(request))
            ctx = MagicMock()
            ctx.__enter__.return_value = resp
            return ctx

        with patch.object(city_data.urllib.request, "urlopen", side_effect=fake_urlopen):
            out = city_data.get_transport_disruptions("city/1 x", lat=1.0, lon=2.0)
        assert captured_url["url"].startswith(
            city_data.get_backend_base_url() + "/municipalities/city%2F1%20x/transports/disruptions?"
        )
        assert out == {"disruptions": [], "note": "Aucune perturbation signalée actuellement."}

    def test_default_backend_url_when_env_missing(self) -> None:
        with patch.dict("os.environ", {"BACKEND_URL": ""}):
            assert city_data.get_backend_base_url() == "http://localhost:3002"
