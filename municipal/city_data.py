"""Données publiées par la mairie : travaux, événements, perturbations transports.

Fonctions read-only utilisées par le chatbot citoyen (/reporting/chat/citoyen).
SQL 100% paramétré ; HTTP backend en fallback gracieux (jamais d'exception).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any
from urllib.parse import quote, urlencode

from municipal.db import get_connection

logger = logging.getLogger("municipall.city_data")

_MAX_LIMIT = 50
_DEFAULT_LIMIT = 10
_DEFAULT_BACKEND_URL = "http://localhost:3002"
_DEFAULT_BACKEND_TIMEOUT_S = 5.0


def _clamp_limit(limit: Any, default: int = _DEFAULT_LIMIT) -> int:
    try:
        n = int(str(limit).strip())
    except (TypeError, ValueError):
        return default
    return max(1, min(_MAX_LIMIT, n))


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_backend_base_url() -> str:
    return (os.environ.get("BACKEND_URL") or _DEFAULT_BACKEND_URL).strip().rstrip("/")


def get_backend_timeout_s() -> float:
    try:
        return float(os.environ.get("BACKEND_TIMEOUT_S", "").strip() or _DEFAULT_BACKEND_TIMEOUT_S)
    except (TypeError, ValueError):
        return _DEFAULT_BACKEND_TIMEOUT_S


def get_construction_works(tenant_id: str, limit: int = 10) -> list[dict[str, Any]]:
    tenant = str(tenant_id or "").strip()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, description, "locationName",
                       "startDate", "endDate", status, "impactType"
                FROM construction_works
                WHERE "tenantId" = %s
                ORDER BY "startDate" DESC
                LIMIT %s
                """,
                (tenant, _clamp_limit(limit)),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "titre": r[0],
                "description": r[1],
                "lieu": r[2],
                "date_debut": r[3].isoformat() if r[3] else None,
                "date_fin": r[4].isoformat() if r[4] else None,
                "statut": r[5],
                "impact": r[6],
            }
        )
    return out


def get_city_events(tenant_id: str, limit: int = 10) -> list[dict[str, Any]]:
    city_id = str(tenant_id or "").strip()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, description, location,
                       start_date, end_date, category
                FROM events
                WHERE city_id = %s
                ORDER BY start_date ASC
                LIMIT %s
                """,
                (city_id, _clamp_limit(limit)),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "titre": r[0],
                "description": r[1],
                "lieu": r[2],
                "date_debut": r[3].isoformat() if r[3] else None,
                "date_fin": r[4].isoformat() if r[4] else None,
                "categorie": r[5],
            }
        )
    return out


def get_transport_disruptions(
    city_id: str,
    lat: float | None = None,
    lon: float | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    city = str(city_id or "").strip()
    la = _coerce_float(lat)
    lo = _coerce_float(lon)
    if la is None or lo is None:
        return {
            "disruptions": [],
            "note": "Position indisponible : impossible de consulter les perturbations transports.",
        }
    if abs(la) > 90 or abs(lo) > 180:
        return {
            "disruptions": [],
            "note": "Coordonnées invalides : impossible de consulter les perturbations transports.",
        }
    params = urlencode({"lat": la, "lon": lo})
    url = (
        f"{get_backend_base_url()}"
        f"/municipalities/{quote(city, safe='')}/transports/disruptions?{params}"
    )
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s or get_backend_timeout_s()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("transport disruptions fetch failed (%s): %s", url, e)
        return {
            "disruptions": [],
            "note": "Service transports momentanément indisponible.",
        }
    lines = data.get("lines") if isinstance(data, dict) else None
    disruptions: list[dict[str, Any]] = []
    for line in lines or []:
        if not isinstance(line, dict) or line.get("status") != "disrupted":
            continue
        disruptions.append(
            {
                "ligne": line.get("lineName"),
                "mode": line.get("mode"),
                "messages": line.get("messages") or [],
            }
        )
    if not disruptions:
        return {"disruptions": [], "note": "Aucune perturbation signalée actuellement."}
    return {"disruptions": disruptions, "note": None}
