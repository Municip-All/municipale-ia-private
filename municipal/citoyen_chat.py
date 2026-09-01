"""Boucle d'agent du chatbot citoyen (/reporting/chat/citoyen).

Réutilise le pattern tool-calling de agent_chat.py : le LLM décide d'appeler
les outils de données mairie (travaux, événements, transports, déchets,
associations, infos pratiques, statut signalement) pour répondre aux
QUESTIONS, ou l'outil create_signalement (pipeline complète) pour un
SIGNALEMENT localisé. Une simple question ne crée jamais de signalement.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from municipal.agent_chat import (
    AGENT_TOOLS,
    _MAX_TEXT_LEN,
    _assistant_tool_message,
    _coerce_bounded_int,
    _coerce_text,
    _parse_arguments,
    _tool_call_payload,
    execute_tool,
)
from municipal.city_data import (
    get_associations,
    get_city_events,
    get_construction_works,
    get_mairie_infos,
    get_report_status,
    get_transport_disruptions,
    get_waste_collection,
)
from municipal.llm_client import chat_completion_tools
from municipal.pipeline import submit_report

logger = logging.getLogger("municipall.citoyen")

_MAX_CITOYEN_STEPS = 4

CITOYEN_SYSTEM_PROMPT = (
    "Tu es l'assistant municipal du chatbot citoyen de Municip'All. "
    "Réponds toujours en français simple, clair et rassurant, maximum 6 phrases. "
    "Décision obligatoire avant de répondre : "
    "1) Si l'utilisateur pose une QUESTION (travaux en cours, transports, événements, "
    "collecte des déchets, associations, horaires ou contacts de la mairie, suivi d'un "
    "signalement, services municipaux), appelle les outils de données de la mairie "
    "(get_construction_works, get_city_events, get_transport_disruptions, "
    "get_waste_collection, get_associations, get_mairie_infos, get_report_status) "
    "puis réponds factuellement à partir des résultats. Si les données renvoyées sont "
    "vides ou accompagnées d'une note, dis-le clairement et oriente vers la rubrique "
    "du site concernée (Travaux, Transports, Déchets & Toilettes, Social & Asso., "
    "Signalements). N'invente jamais d'information. "
    "2) Si l'utilisateur décrit un SIGNALEMENT (problème localisé à faire traiter par "
    "la mairie : voirie, éclairage, propreté, espaces verts, équipement cassé…), appelle "
    "l'outil create_signalement avec le texte du problème, puis confirme le traitement "
    "d'après le résultat (statut, catégorie, service) et communique le numéro de "
    "signalement en précisant qu'il permettra d'en suivre l'avancement. Si l'utilisateur "
    "ne donne pas de lieu ou de description assez précise, demande une précision avant "
    "d'enregistrer. "
    "Tu peux utiliser smart_analyzer, smart_route et duplicate_finder pour analyser "
    "un texte de signalement avant l'enregistrement. "
    "JAMAIS appeler create_signalement pour une simple question."
)

_CITOYEN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_construction_works",
            "description": (
                "Liste les travaux publiés par la mairie (titre, description, lieu, "
                "dates de début/fin, statut, impact circulation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de résultats (1-20, défaut 10)",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_events",
            "description": (
                "Liste les événements municipaux publiés (titre, description, lieu, "
                "dates, catégorie)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de résultats (1-20, défaut 10)",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transport_disruptions",
            "description": (
                "Perturbations en cours sur les transports en commun à proximité "
                "(données temps réel du backend de la ville)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude de la position (optionnelle)",
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude de la position (optionnelle)",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_signalement",
            "description": (
                "Enregistre un signalement citoyen et lance le pipeline complète "
                "(spam, sentiment, catégorie, doublon). À utiliser UNIQUEMENT quand "
                "l'utilisateur décrit un problème localisé à signaler, JAMAIS pour "
                "une simple question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texte": {
                        "type": "string",
                        "description": "Texte du signalement de l'utilisateur",
                        "maxLength": _MAX_TEXT_LEN,
                    }
                },
                "required": ["texte"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_waste_collection",
            "description": (
                "Calendrier de collecte des déchets publié par la mairie "
                "(types de collecte, jours, horaires)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de services (1-20, défaut 10)",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_associations",
            "description": (
                "Associations, groupes citoyens et initiatives locales recensés "
                "par la commune (nom, catégorie, description, contacts)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de résultats (1-20, défaut 10)",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mairie_infos",
            "description": (
                "Informations pratiques de la mairie : horaires d'ouverture, adresse, "
                "site web, nom du maire, numéros utiles et liens pratiques."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_report_status",
            "description": (
                "Statut d'avancement d'un signalement (statut, catégorie, service en "
                "charge, dates). Demander le numéro du signalement à l'utilisateur "
                "s'il ne l'a pas fourni."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "numero": {
                        "type": "integer",
                        "description": "Numéro du signalement (ex : 42)",
                    }
                },
                "required": ["numero"],
                "additionalProperties": False,
            },
        },
    },
] + [t for t in AGENT_TOOLS if t["function"]["name"] in ("smart_analyzer", "smart_route", "duplicate_finder")]


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def execute_citoyen_tool(
    name: str,
    arguments: dict[str, Any],
    user_id: str,
    tenant_id: str,
) -> Any:
    if name == "create_signalement":
        return submit_report(user_id, _coerce_text(arguments.get("texte")), tenant_id=tenant_id)
    if name == "get_construction_works":
        return get_construction_works(
            tenant_id,
            limit=_coerce_bounded_int(arguments.get("limit"), 10, 1, 20),
        )
    if name == "get_city_events":
        return get_city_events(
            tenant_id,
            limit=_coerce_bounded_int(arguments.get("limit"), 10, 1, 20),
        )
    if name == "get_transport_disruptions":
        return get_transport_disruptions(
            tenant_id,
            lat=_coerce_float(arguments.get("lat")),
            lon=_coerce_float(arguments.get("lon")),
        )
    if name == "get_waste_collection":
        return get_waste_collection(
            tenant_id,
            limit=_coerce_bounded_int(arguments.get("limit"), 10, 1, 20),
        )
    if name == "get_associations":
        return get_associations(
            tenant_id,
            limit=_coerce_bounded_int(arguments.get("limit"), 10, 1, 20),
        )
    if name == "get_mairie_infos":
        return get_mairie_infos(tenant_id)
    if name == "get_report_status":
        return get_report_status(arguments.get("numero"), tenant_id)
    return execute_tool(name, arguments, tenant_id=tenant_id)


def _report_confirmation(report: dict[str, Any]) -> str:
    if report.get("is_spam"):
        return (
            "Votre message a été enregistré pour modération (contenu signalé comme non conforme). "
            "Les signalements publics doivent concerner la vie municipale."
        )
    if report.get("status") == "Doublon":
        return (
            "Nous avons détecté un signalement très proche du vôtre déjà en cours. "
            "Votre demande est classée en doublon et rattachée au dossier existant."
        )
    return (
        f"Votre signalement a bien été enregistré. Il concerne la thématique "
        f"« {report.get('category')} » et sera transmis au service : {report.get('municipal_service')}."
    )


def run_citoyen_chat(message: str, user_id: str, tenant_id: str) -> dict[str, Any]:
    q = (message or "").strip()[:_MAX_TEXT_LEN]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CITOYEN_SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    created_report: dict[str, Any] | None = None
    tools_used: list[str] = []
    try:
        for _ in range(_MAX_CITOYEN_STEPS):
            llm_message = chat_completion_tools(messages, tools=_CITOYEN_TOOLS, temperature=0.25)
            tool_calls = list(getattr(llm_message, "tool_calls", None) or [])
            if not tool_calls:
                reply = (getattr(llm_message, "content", None) or "").strip()
                if not reply:
                    raise RuntimeError("réponse_vide")
                return {"reply": reply, "created_report": created_report, "tools_used": tools_used}
            messages.append(_assistant_tool_message(llm_message, tool_calls))
            for i, tool_call in enumerate(tool_calls):
                payload = _tool_call_payload(tool_call, i)
                name = payload["function"]["name"]
                arguments = _parse_arguments(payload["function"]["arguments"])
                try:
                    result = execute_citoyen_tool(name, arguments, user_id=user_id, tenant_id=tenant_id)
                    content = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    logger.warning("citoyen tool %s failed: %s", name, e)
                    content = json.dumps({"error": str(e)}, ensure_ascii=False)
                else:
                    if name == "create_signalement" and isinstance(result, dict):
                        created_report = result
                messages.append(
                    {"role": "tool", "tool_call_id": payload["id"], "content": content}
                )
                tools_used.append(name)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tu as atteint la limite d'appels d'outils. "
                    "Synthétise maintenant une réponse finale en français à partir des résultats obtenus."
                ),
            }
        )
        final = chat_completion_tools(messages, tools=None, temperature=0.25)
        reply = (getattr(final, "content", None) or "").strip()
        if not reply:
            raise RuntimeError("réponse_vide")
        return {"reply": reply, "created_report": created_report, "tools_used": tools_used}
    except Exception as e:
        if created_report is not None:
            logger.warning("citoyen chat degraded after report creation: %s", e)
            return {
                "reply": _report_confirmation(created_report),
                "created_report": created_report,
                "tools_used": tools_used,
                "degraded": True,
            }
        raise
