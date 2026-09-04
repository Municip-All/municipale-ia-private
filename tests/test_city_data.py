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

    def test_missing_coordinates_falls_back_to_city_center(self) -> None:
        with patch.object(
            city_data.urllib.request, "urlopen", side_effect=OSError("no network")
        ) as mock_urlopen:
            out = city_data.get_transport_disruptions("le-kremlin-bicetre")
        assert mock_urlopen.call_count == 1
        assert "lat=48.812" in mock_urlopen.call_args[0][0].full_url
        assert out["disruptions"] == []
        assert out["note"]

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


def _fake_conn_one(row, capture: dict) -> MagicMock:
    cur = MagicMock()
    conn = MagicMock()

    def execute(query, params=None):
        capture["query"] = query
        capture["params"] = params
        cur.fetchone.return_value = row

    cur.execute.side_effect = execute
    conn.cursor.return_value.__enter__.return_value = cur
    return conn


class TestGetWasteCollection:
    def test_services_mapped_with_french_days(self) -> None:
        config = {
            "services": [
                {"type": "Ordures ménagères", "days": [1, 4], "time": "05:30"},
                {"type": "Tri sélectif", "days": [3], "time": "06:00"},
            ]
        }
        capture: dict = {}
        conn = _fake_conn_one((json.dumps(config),), capture)
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_waste_collection("ville-a")
        assert 'WHERE id = %s' in capture["query"]
        assert capture["params"] == ("ville-a",)
        assert out["note"] is None
        assert out["services"][0] == {
            "service": "Ordures ménagères",
            "jours": "lundi, jeudi",
            "heure": "05:30",
        }
        assert out["services"][1]["jours"] == "mercredi"

    def test_missing_config_returns_note(self) -> None:
        conn = _fake_conn_one((None,), {})
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_waste_collection("ville-a")
        assert out == {"services": [], "note": city_data.get_waste_collection.__doc__ or out["note"]}
        assert out["services"] == []
        assert "Déchets & Toilettes" in out["note"]


class TestGetAssociations:
    def test_associations_mapped(self) -> None:
        items = [
            {
                "name": "Comité de quartier",
                "category": "association",
                "description": "Actions locales",
                "address": "1 rue Centrale",
                "contactEmail": "contact@asso.fr",
                "contactPhone": "0102030405",
                "website": "https://exemple.fr",
            }
        ]
        conn = _fake_conn_one((json.dumps(items),), {})
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_associations("ville-a")
        assert out["note"] is None
        assert out["associations"][0]["nom"] == "Comité de quartier"
        assert out["associations"][0]["telephone"] == "0102030405"

    def test_empty_returns_note(self) -> None:
        conn = _fake_conn_one((None,), {})
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_associations("ville-a")
        assert out["associations"] == []
        assert "Social & Asso." in out["note"]


class TestGetMairieInfos:
    def test_profile_numbers_and_links(self) -> None:
        profile = {
            "openingHours": "Lun-Ven 8h30-17h",
            "address": "1 place de la Mairie",
            "website": "https://ville.fr",
            "mayorName": "Sophie Martin",
        }
        numbers = [{"label": "Police", "phone": "17"}]
        links = [{"label": "Site officiel", "url": "https://ville.fr"}]
        conn = _fake_conn_one(
            (json.dumps(profile), json.dumps(numbers), json.dumps(links)), {}
        )
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_mairie_infos("ville-a")
        assert out["horaires"] == "Lun-Ven 8h30-17h"
        assert out["maire"] == "Sophie Martin"
        assert out["numeros_utiles"] == [{"label": "Police", "telephone": "17"}]
        assert out["note"] is None

    def test_empty_profile_returns_note(self) -> None:
        conn = _fake_conn_one((None, None, None), {})
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_mairie_infos("ville-a")
        assert "Aucune information pratique" in out["note"]


class TestGetReportStatus:
    def test_found_maps_public_fields_only(self) -> None:
        capture: dict = {}
        conn = _fake_conn_one(
            (
                12,
                "En attente",
                "Éclairage public",
                "Services techniques",
                None,
                False,
                datetime(2026, 8, 28, 9, 0),
                datetime(2026, 8, 30, 10, 0),
            ),
            capture,
        )
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_report_status("12", "ville-a")
        assert "tenant_id = %s" in capture["query"]
        assert capture["params"] == (12, "ville-a")
        assert out["trouve"] is True
        assert out["numero"] == 12
        assert out["statut"] == "En attente"
        assert out["service_en_charge"] == "Services techniques"
        assert "description" not in out

    def test_not_found_returns_note(self) -> None:
        conn = _fake_conn_one(None, {})
        with patch.object(city_data, "get_connection") as fake:
            fake.return_value.__enter__.return_value = conn
            out = city_data.get_report_status(999, "ville-a")
        assert out["trouve"] is False
        assert "Aucun signalement trouvé" in out["note"]

    def test_invalid_id_never_queries(self) -> None:
        with patch.object(city_data, "get_connection") as fake:
            out = city_data.get_report_status("abc", "ville-a")
        fake.assert_not_called()
        assert out["trouve"] is False

    def test_out_of_bounds_id_rejected(self) -> None:
        out = city_data.get_report_status(-5, "ville-a")
        assert out["trouve"] is False
