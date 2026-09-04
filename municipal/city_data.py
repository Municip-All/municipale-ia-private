"""Données publiées par la mairie : travaux, événements, perturbations transports,
collecte des déchets, associations, infos pratiques mairie, statut d'un signalement.

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


_CITY_CENTERS: dict[str, tuple[float, float]] = {
    "le-kremlin-bicetre": (48.8120, 2.3590),
}
_DEFAULT_CITY_CENTER = (48.8566, 2.3522)


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
        center = _CITY_CENTERS.get(city) or _DEFAULT_CITY_CENTER
        la, lo = center
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


_WASTE_DAYS_FR = {
    0: "dimanche",
    1: "lundi",
    2: "mardi",
    3: "mercredi",
    4: "jeudi",
    5: "vendredi",
    6: "samedi",
}


def _load_city_json(city_id: str, columns: list[str]) -> dict[str, Any] | None:
    city = str(city_id or "").strip()
    if not city:
        return None
    cols = ", ".join(f'"{c}"' for c in columns)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM cities WHERE id = %s",
                (city,),
            )
            row = cur.fetchone()
    if not row:
        return None
    out: dict[str, Any] = {c: row[i] for i, c in enumerate(columns)}
    for k, v in out.items():
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except (TypeError, ValueError):
                pass
    return out


def get_waste_collection(city_id: str, limit: int = 15) -> dict[str, Any]:
    data = _load_city_json(city_id, ["waste_config"]) or {}
    config = data.get("waste_config") or {}
    services = config.get("services") if isinstance(config, dict) else None
    out: list[dict[str, Any]] = []
    for s in (services or [])[:_clamp_limit(limit)]:
        if not isinstance(s, dict):
            continue
        days = s.get("days")
        jours = ", ".join(
            _WASTE_DAYS_FR.get(d, str(d)) for d in days if isinstance(d, int)
        ) if isinstance(days, list) else None
        out.append(
            {
                "service": s.get("type"),
                "jours": jours,
                "heure": s.get("time"),
            }
        )
    if not out:
        return {
            "services": [],
            "note": (
                "Aucun calendrier de collecte publié pour cette commune. "
                "Les points de collecte et sanitaires publics géolocalisés sont "
                "consultables dans la rubrique Déchets & Toilettes du site."
            ),
        }
    return {"services": out, "note": None}


def get_associations(city_id: str, limit: int = 15) -> dict[str, Any]:
    data = _load_city_json(city_id, ["associations"]) or {}
    items = data.get("associations") or []
    out: list[dict[str, Any]] = []
    for a in items[:_clamp_limit(limit)]:
        if not isinstance(a, dict):
            continue
        out.append(
            {
                "nom": a.get("name"),
                "categorie": a.get("category"),
                "description": a.get("description"),
                "adresse": a.get("address"),
                "email": a.get("contactEmail"),
                "telephone": a.get("contactPhone"),
                "site_web": a.get("website"),
            }
        )
    if not out:
        return {
            "associations": [],
            "note": (
                "Aucune association publiée pour cette commune. "
                "La rubrique Social & Asso. du site recense les initiatives locales."
            ),
        }
    return {"associations": out, "note": None}


def get_mairie_infos(city_id: str) -> dict[str, Any]:
    data = _load_city_json(city_id, ["public_profile", "useful_numbers", "useful_links"]) or {}
    profile = data.get("public_profile") or {}
    numbers = data.get("useful_numbers") or []
    links = data.get("useful_links") or []
    out: dict[str, Any] = {
        "horaires": profile.get("openingHours"),
        "adresse": profile.get("address"),
        "site_web": profile.get("website"),
        "maire": profile.get("mayorName"),
        "numeros_utiles": [
            {"label": n.get("label"), "telephone": n.get("phone")}
            for n in numbers if isinstance(n, dict)
        ][:10],
        "liens_utiles": [
            {"label": l.get("label"), "url": l.get("url")}
            for l in links if isinstance(l, dict)
        ][:10],
        "note": None,
    }
    if not any([out["horaires"], out["adresse"], out["numeros_utiles"], out["liens_utiles"]]):
        out["note"] = "Aucune information pratique publiée pour cette commune."
    return out


def get_report_status(report_id: Any, tenant_id: str) -> dict[str, Any]:
    try:
        rid = int(str(report_id).strip())
    except (TypeError, ValueError):
        return {"trouve": False, "note": "Numéro de signalement invalide."}
    if report_id is None or rid < 1 or rid > 2_000_000_000:
        return {"trouve": False, "note": "Numéro de signalement invalide."}
    tenant = str(tenant_id or "").strip()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, category, municipal_service,
                       duplicate_of_id, is_spam, created_at, updated_at
                FROM reports
                WHERE id = %s AND tenant_id = %s
                LIMIT 1
                """,
                (rid, tenant),
            )
            row = cur.fetchone()
    if not row:
        return {
            "trouve": False,
            "note": (
                "Aucun signalement trouvé avec ce numéro. "
                "Le numéro figure dans la confirmation envoyée après la création."
            ),
        }
    return {
        "trouve": True,
        "numero": row[0],
        "statut": row[1],
        "categorie": row[2],
        "service_en_charge": row[3],
        "doublon_de": row[4],
        "signale_comme_spam": bool(row[5]),
        "cree_le": row[6].isoformat() if row[6] else None,
        "maj_le": row[7].isoformat() if row[7] else None,
    }
