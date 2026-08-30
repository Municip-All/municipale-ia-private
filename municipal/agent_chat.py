from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from municipal.analyzer import smart_analyzer
from municipal.db import (
    REPORT_GROUP_VALUES,
    REPORT_ORDER_VALUES,
    count_reports,
    query_reports,
    top_urgent_by_sentiment,
)
from municipal.duplicate import duplicate_finder
from municipal.embeddings import embed_one
from municipal.llm_client import chat_completion_tools
from municipal.router import smart_route

logger = logging.getLogger("municipall.agent")

_MAX_AGENT_STEPS = 4
_MAX_TEXT_LEN = 5000
_MAX_QUESTION_LEN = 2000

_REPORT_STATUS_VALUES = ["En attente", "En cours", "Résolu", "Doublon", "Rejeté", "Spam"]

AGENT_SYSTEM_PROMPT = (
    "Tu es l'assistant IA professionnel des agents de mairie du back-office Municip'All. "
    "Réponds toujours en français : synthétique, opérationnel, chiffres à l'appui. "
    "Tu disposes d'outils : smart_analyzer (spam/sentiment/urgence d'un texte), "
    "smart_route (catégorie municipale et service compétent), "
    "duplicate_finder (recherche de doublon sémantique), "
    "top_urgent_by_sentiment (signalements ouverts les plus urgents), "
    "query_reports (liste filtrée des signalements : statut, catégorie, période, tri), "
    "count_reports (comptages agrégés par catégorie ou statut). "
    "Appelle un ou plusieurs outils dès que la question porte sur des données réelles des signalements. "
    "Appuie ta réponse sur les résultats des outils en citant les IDs, statuts, catégories et scores. "
    "Termine toujours par une action concrète proposée : un signalement à traiter, à requalifier ou à clôturer. "
    "Si aucun outil n'est pertinent, réponds directement sans en appeler."
)

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "smart_analyzer",
            "description": (
                "Analyse un texte de signalement citoyen : détection de spam, "
                "score de sentiment (-1 très négatif à 1 très positif), urgence et ton."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte du signalement à analyser",
                        "maxLength": _MAX_TEXT_LEN,
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smart_route",
            "description": (
                "Détermine la catégorie municipale (8 catégories) et le service "
                "compétent pour un texte de signalement, avec un score de confiance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte du signalement à router",
                        "maxLength": _MAX_TEXT_LEN,
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duplicate_finder",
            "description": (
                "Cherche dans la base le signalement existant le plus proche "
                "sémantiquement du texte fourni et indique si c'est un doublon."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Texte du signalement à comparer",
                        "maxLength": _MAX_TEXT_LEN,
                    },
                    "exclude_report_id": {
                        "type": "integer",
                        "description": "ID d'un signalement à exclure de la recherche",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Seuil de similarité cosinus entre 0 et 1 (défaut 0.85)",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_urgent_by_sentiment",
            "description": (
                "Liste les signalements ouverts les plus urgents, triés par sentiment "
                "le plus négatif, sur une période donnée en jours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Période de recherche en jours (1-90, défaut 7)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de résultats (1-20, défaut 3)",
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
            "name": "query_reports",
            "description": (
                "Liste les signalements de la base avec filtres combinables : statut, catégorie, "
                "période en jours, tri et nombre de résultats. Pour toute question du type "
                "« quels travaux sont en cours », « signalements récents de la voirie », "
                "« derniers signalements rejetés »."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Statut des signalements recherchés",
                        "enum": _REPORT_STATUS_VALUES,
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Catégorie municipale exacte "
                            "(ex : Voirie, Éclairage public, Espaces verts)"
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": "Période de recherche en jours (1-365, défaut 30)",
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Tri des résultats (défaut created_at_desc)",
                        "enum": REPORT_ORDER_VALUES,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre maximum de résultats (1-50, défaut 20)",
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
            "name": "count_reports",
            "description": (
                "Compte les signalements agrégés par groupe : catégorie, statut, service municipal "
                "ou catégorie IA. Pour les questions du type « combien de signalements par catégorie » "
                "ou « répartition par statut »."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "description": "Champ d'agrégation (défaut status)",
                        "enum": REPORT_GROUP_VALUES,
                    },
                    "days": {
                        "type": "integer",
                        "description": "Période optionnelle en jours (1-365)",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


class AgentChatOut(BaseModel):
    answer: str
    top_reports: list[dict[str, Any]] = Field(default_factory=list)
    analyses: list[dict[str, Any]] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    fallback: bool = False


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()[:_MAX_TEXT_LEN]


def _coerce_bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _coerce_threshold(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, t))


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "smart_analyzer":
        result = smart_analyzer(_coerce_text(arguments.get("text")))
        result.pop("embedding", None)
        return result
    if name == "smart_route":
        return smart_route(_coerce_text(arguments.get("text")))
    if name == "duplicate_finder":
        text = _coerce_text(arguments.get("text"))
        embedding = embed_one(text)
        return duplicate_finder(
            embedding,
            exclude_report_id=_coerce_optional_int(arguments.get("exclude_report_id")),
            threshold=_coerce_threshold(arguments.get("threshold")),
        )
    if name == "top_urgent_by_sentiment":
        return top_urgent_by_sentiment(
            days=_coerce_bounded_int(arguments.get("days"), 7, 1, 90),
            limit=_coerce_bounded_int(arguments.get("limit"), 3, 1, 20),
        )
    if name == "query_reports":
        return query_reports(
            status=arguments.get("status"),
            category=arguments.get("category"),
            days=_coerce_bounded_int(arguments.get("days"), 30, 1, 365),
            order_by=str(arguments.get("order_by") or "created_at_desc"),
            limit=_coerce_bounded_int(arguments.get("limit"), 20, 1, 50),
        )
    if name == "count_reports":
        days = _coerce_optional_int(arguments.get("days"))
        return count_reports(
            group_by=str(arguments.get("group_by") or "status"),
            days=max(1, min(365, days)) if days is not None else None,
        )
    raise ValueError(f"outil_inconnu:{name}")


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_call_payload(tool_call: Any, index: int) -> dict[str, Any]:
    function = getattr(tool_call, "function", None)
    name = str(getattr(function, "name", "") or "")
    arguments = getattr(function, "arguments", "{}")
    call_id = getattr(tool_call, "id", None) or f"call_{index}"
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)},
    }


def _assistant_tool_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [
            _tool_call_payload(tool_call, i) for i, tool_call in enumerate(tool_calls)
        ],
    }


def _wants_urgent_top3(q: str) -> bool:
    return bool(
        re.search(
            r"(3|trois).{0,40}(probl[eè]me|signalement|urgent|urgence)|"
            r"urgent|sentiment|cette semaine|semaine",
            (q or "").lower(),
        )
    )


_FALLBACK_STATUS_PATTERNS: list[tuple[str, str]] = [
    (r"en cours|travaux", "En cours"),
    (r"en attente|non trait", "En attente"),
    (r"résolu|resolu|clôtur|clotur", "Résolu"),
    (r"rejet", "Rejeté"),
    (r"doublon", "Doublon"),
    (r"spam|indésirable", "Spam"),
]

_FALLBACK_GROUP_LABELS: dict[str, str] = {
    "category": "catégorie",
    "status": "statut",
    "municipal_service": "service municipal",
    "ai_category": "catégorie IA",
}


def _fallback_status_for(q: str) -> str | None:
    ql = (q or "").lower()
    for pattern, status in _FALLBACK_STATUS_PATTERNS:
        if re.search(pattern, ql):
            return status
    return None


def _wants_category_breakdown(q: str) -> bool:
    return bool(
        re.search(r"par cat[ée]gorie|r[ée]partition|combien", (q or "").lower())
    )


def _format_query_rows(rows: list[dict[str, Any]], status: str) -> str:
    if not rows:
        return (
            f"Aucun signalement « {status} » sur la période récente. "
            "Action suggérée : élargir la période ou consulter le tableau complet."
        )
    lines = []
    for r in rows[:10]:
        content = str(r.get("content") or "")
        created = str(r.get("created_at") or "")[:10]
        lines.append(
            f"- #{r.get('id')} [{r.get('category')}] "
            f"{content[:120]}{'…' if len(content) > 120 else ''} ({created})"
        )
    return (
        f"Signalements « {status} » ({len(rows)} résultat(s)) :\n"
        + "\n".join(lines)
        + "\nAction suggérée : traiter en priorité les signalements au sentiment le plus négatif."
    )


def _format_counts(rows: list[dict[str, Any]], group_by: str) -> str:
    if not rows:
        return (
            "Aucun signalement en base sur la période demandée. "
            "Action suggérée : vérifier la période ou la connexion à la base."
        )
    total = sum(int(r.get("count") or 0) for r in rows)
    lines = [f"- {r.get('group_key')} : {r.get('count')}" for r in rows[:12]]
    label = _FALLBACK_GROUP_LABELS.get(group_by, group_by)
    return (
        f"Répartition des signalements par {label} ({total} au total) :\n"
        + "\n".join(lines)
        + "\nAction suggérée : traiter d'abord le groupe le plus fourni."
    )


def _format_urgent_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(rows, 1):
        content = r.get("content") or ""
        lines.append(
            f"{i}. [{r.get('category')}] score {float(r.get('sentiment_score') or 0):.2f} — "
            f"{content[:160]}{'…' if len(content) > 160 else ''}"
        )
    return (
        "Les 3 signalements ouverts les plus urgents (sentiment le plus négatif, 7 jours) :\n"
        + "\n".join(lines)
    )


def mairie_fallback_chat(question: str) -> AgentChatOut:
    if _wants_urgent_top3(question):
        try:
            rows = top_urgent_by_sentiment(days=7, limit=3)
        except Exception as e:
            logger.warning("agent fallback db unreachable: %s", e)
            return AgentChatOut(
                answer="La base de signalements est momentanément indisponible. Réessayez plus tard.",
                fallback=True,
            )
        if rows:
            return AgentChatOut(answer=_format_urgent_rows(rows), top_reports=rows, fallback=True)
        return AgentChatOut(
            answer=(
                "Aucun signalement ouvert sur les 7 derniers jours ou base vide. "
                "Vérifiez la période ou consultez le système complet."
            ),
            fallback=True,
        )
    if _wants_category_breakdown(question):
        try:
            counts = count_reports(group_by="category")
        except Exception as e:
            logger.warning("agent fallback db unreachable: %s", e)
            return AgentChatOut(
                answer="La base de signalements est momentanément indisponible. Réessayez plus tard.",
                fallback=True,
            )
        return AgentChatOut(answer=_format_counts(counts, "category"), fallback=True)
    status = _fallback_status_for(question)
    if status:
        try:
            rows = query_reports(status=status, days=30, order_by="created_at_desc", limit=10)
        except Exception as e:
            logger.warning("agent fallback db unreachable: %s", e)
            return AgentChatOut(
                answer="La base de signalements est momentanément indisponible. Réessayez plus tard.",
                fallback=True,
            )
        return AgentChatOut(
            answer=_format_query_rows(rows, status),
            top_reports=rows,
            fallback=True,
        )
    return AgentChatOut(
        answer=(
            "L'assistant IA n'est pas disponible actuellement. "
            "Posez par exemple : « Quels sont les 3 problèmes les plus urgents "
            "basés sur le sentiment des citoyens cette semaine ? »"
        ),
        fallback=True,
    )


def run_agent_chat(question: str) -> AgentChatOut:
    q = (question or "").strip()[:_MAX_QUESTION_LEN]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    analyses: list[dict[str, Any]] = []
    top_reports: list[dict[str, Any]] = []
    tools_used: list[str] = []
    try:
        for _ in range(_MAX_AGENT_STEPS):
            message = chat_completion_tools(messages, tools=AGENT_TOOLS, temperature=0.2)
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls:
                answer = (getattr(message, "content", None) or "").strip()
                if not answer:
                    return mairie_fallback_chat(q)
                return AgentChatOut(
                    answer=answer,
                    top_reports=top_reports,
                    analyses=analyses,
                    tools_used=tools_used,
                )
            messages.append(_assistant_tool_message(message, tool_calls))
            for i, tool_call in enumerate(tool_calls):
                payload = _tool_call_payload(tool_call, i)
                name = payload["function"]["name"]
                arguments = _parse_arguments(payload["function"]["arguments"])
                trace: dict[str, Any] = {"tool": name, "arguments": arguments}
                try:
                    result = execute_tool(name, arguments)
                    trace["result"] = result
                    content = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    logger.warning("agent tool %s failed: %s", name, e)
                    trace["error"] = str(e)
                    content = json.dumps({"error": str(e)}, ensure_ascii=False)
                messages.append(
                    {"role": "tool", "tool_call_id": payload["id"], "content": content}
                )
                analyses.append(trace)
                tools_used.append(name)
                if isinstance(trace.get("result"), list) and name in (
                    "top_urgent_by_sentiment",
                    "query_reports",
                ):
                    top_reports = trace["result"]
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tu as atteint la limite d'appels d'outils. "
                    "Synthétise maintenant une réponse finale en français à partir des résultats obtenus."
                ),
            }
        )
        final = chat_completion_tools(messages, tools=None, temperature=0.2)
        answer = (getattr(final, "content", None) or "").strip()
        if not answer:
            return mairie_fallback_chat(q)
        return AgentChatOut(
            answer=answer,
            top_reports=top_reports,
            analyses=analyses,
            tools_used=tools_used,
        )
    except Exception as e:
        logger.warning("agent chat degraded to mairie fallback: %s", e)
        return mairie_fallback_chat(q)
