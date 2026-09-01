from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from municipal.db import get_conninfo, top_urgent_by_sentiment, enrich_report
from municipal.llm_client import chat_completion, llm_configured
from municipal.pipeline import submit_report
from municipal.analyzer import smart_analyzer
from municipal.router import smart_route
from municipal.duplicate import duplicate_finder
from municipal.rate_limit import limiter
from municipal.agent_chat import AgentChatOut, mairie_fallback_chat, run_agent_chat
from municipal.citoyen_chat import run_citoyen_chat

logger = logging.getLogger("municipall.reporting")

router = APIRouter(prefix="/reporting", tags=["reporting"])


def _citoyen_template(r: dict[str, Any]) -> str:
    cat = r["category"]
    svc = r["municipal_service"]
    if r.get("is_spam"):
        return (
            "Votre message a été enregistré pour modération (contenu signalé comme non conforme). "
            "Les signalements publics doivent concerner la vie municipale."
        )
    if r["status"] == "Doublon":
        return (
            f"Nous avons détecté un signalement très proche du vôtre déjà en cours. "
            f"Votre demande est classée en doublon et rattachée au dossier existant (thématique : {cat})."
        )
    return (
        f"Votre demande est bien prise en compte. Elle concerne la thématique « {cat} » "
        f"et sera transmise au service : {svc}."
    )


def _citoyen_llm(r: dict[str, Any], user_message: str) -> str:
    ctx = {
        "message_utilisateur": user_message,
        "traitement": {
            "statut": r.get("status"),
            "categorie": r.get("category"),
            "service_municipal": r.get("municipal_service"),
            "score_sentiment": r.get("sentiment_score"),
            "spam": r.get("is_spam"),
            "signalement_doublon_de": r.get("duplicate_of_id"),
        },
    }
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "Tu es l'assistant municipale de la plateforme Municip'All. "
                    "Réponds en français, avec un ton rassurant, clair et professionnel. "
                    "Maximum 4 phrases. Ne promets pas de délais précis. "
                    "Adapte ton message au statut (spam, doublon, prise en compte normale) décrit dans le contexte."
                ),
            },
            {
                "role": "user",
                "content": "Contexte JSON (données internes, ne pas citer d'ID techniques si inutile) :\n"
                + json.dumps(ctx, ensure_ascii=False),
            },
        ],
        temperature=0.25,
    )


def _mairie_llm(query: str, top_reports: list[dict[str, Any]]) -> str:
    if top_reports:
        ctx = "Données (signalements ouverts, urgence liée au sentiment) :\n" + json.dumps(
            top_reports, ensure_ascii=False, default=str
        )
        u = f"Question : {query}\n\n{ctx}\n\nRéponds en français, de façon synthétique (liste ou paragraphe court)."
    else:
        u = (
            f"Question d'un agent de mairie (dashboard signalements) : {query}\n"
            "Réponds en français, de manière pratique. "
            "Si des données chiffrées manquent, indique qu'il faut consulter le système ou préciser le périmètre."
        )
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "Tu es l'assistant d'analyse du back-office municipal pour Municip'All. "
                    "Style sobre, en français, orienté opérationnel."
                ),
            },
            {"role": "user", "content": u},
        ],
        temperature=0.2,
    )


def _mairie_fallback(query: str, top_reports: list[dict[str, Any]]) -> str:
    if not top_reports:
        return (
            "Aucun signalement ouvert sur les 7 derniers jours ou base vide. "
            "Vérifiez la période ou consultez le système complet."
        )
    lines = []
    for i, r in enumerate(top_reports, 1):
        lines.append(
            f"{i}. [{r['category']}] score {r['sentiment_score']:.2f} — "
            f"{(r['content'] or '')[:160]}{'…' if len(r.get('content') or '') > 160 else ''}"
        )
    return "Les 3 signalements ouverts les plus urgents (sentiment le plus négatif, 7 jours) :\n" + "\n".join(lines)


def _wants_urgent_top3(q: str) -> bool:
    return bool(
        re.search(
            r"(3|trois).{0,40}(probl[eè]me|signalement|urgent|urgence)|"
            r"urgent|sentiment|cette semaine|semaine",
            (q or "").lower(),
        )
    )


class EnrichIn(BaseModel):
    report_id: int = Field(..., description="ID du signalement (INT, backend NestJS)")
    tenant_id: str = Field(..., description="Tenant ID (backend)", max_length=128)
    user_id: int | None = Field(None, description="User ID numérique")
    content: str = Field(..., description="Texte du signalement", max_length=5000)
    lat: float | None = Field(None)
    lon: float | None = Field(None)


class EnrichOut(BaseModel):
    category: str
    municipal_service: str
    sentiment_score: float
    is_spam: bool
    duplicate_of_id: int | None = None
    ai_confidence: float
    ai_status: str


@router.post("/enrich", response_model=EnrichOut)
@limiter.limit("30/minute")
def enrich_existing(request: Request, payload: EnrichIn) -> EnrichOut:
    try:
        get_conninfo()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    a = smart_analyzer(payload.content, str(payload.user_id) if payload.user_id else None)
    r = smart_route(payload.content)
    dup_id: int | None = None
    dup_snapshot: dict[str, Any] | None = None

    if a.get("is_spam"):
        ai_status = "Spam"
        dup_snapshot = {"skipped": True, "reason": "spam_detecte"}
    else:
        d = duplicate_finder(
            a["embedding"],
            exclude_report_id=payload.report_id,
            threshold=None,
        )
        dup_snapshot = {
            key: d.get(key)
            for key in (
                "found", "is_duplicate", "match_id",
                "match_status", "best_similarity", "message",
            )
            if key in d
        }
        if d.get("is_duplicate") and d.get("match_id"):
            dup_id = d["match_id"]
            ai_status = "Doublon"
        else:
            ai_status = "En attente"

    enrich_report(
        report_id=payload.report_id,
        tenant_id=payload.tenant_id,
        content=payload.content,
        category=r["category"],
        municipal_service=r["municipal_service"],
        sentiment_score=float(a["sentiment_score"]),
        embedding=a["embedding"],
        is_spam=bool(a["is_spam"]),
        duplicate_of_id=dup_id,
    )

    return EnrichOut(
        category=r["category"],
        municipal_service=r["municipal_service"],
        sentiment_score=float(a["sentiment_score"]),
        is_spam=bool(a["is_spam"]),
        duplicate_of_id=dup_id,
        ai_confidence=float(r["confidence"]),
        ai_status=ai_status,
    )


class SubmitIn(BaseModel):
    user_id: str = Field(..., description="UUID utilisateur", max_length=128)
    content: str = Field(..., description="Texte du signalement", max_length=5000)
    tenant_id: str = Field("ia-pipeline", description="Tenant ID", max_length=128)


class SubmitOut(BaseModel):
    report_id: str
    status: str
    category: str
    municipal_service: str
    sentiment_score: float
    is_spam: bool
    duplicate_of_id: Optional[str] = None


@router.post("/submit", response_model=SubmitOut)
@limiter.limit("30/minute")
def api_submit(request: Request, payload: SubmitIn) -> SubmitOut:
    try:
        get_conninfo()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    try:
        r = submit_report(payload.user_id, payload.content, tenant_id=payload.tenant_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return SubmitOut(
        report_id=r["report_id"],
        status=r["status"],
        category=r["category"],
        municipal_service=r["municipal_service"],
        sentiment_score=float(r["sentiment_score"]),
        is_spam=bool(r["is_spam"]),
        duplicate_of_id=r.get("duplicate_of_id"),
    )


class CitoyenChatIn(BaseModel):
    user_id: str = Field(..., max_length=128)
    message: str = Field(..., max_length=5000)
    tenant_id: str = Field("ia-pipeline", description="Tenant ID", max_length=128)


class CitoyenChatOut(BaseModel):
    reply: str
    category: str
    municipal_service: str
    sentiment_score: float
    reassured: bool = True


@router.post("/chat/citoyen", response_model=CitoyenChatOut)
@limiter.limit("10/minute")
def chat_citoyen(request: Request, payload: CitoyenChatIn) -> CitoyenChatOut:
    try:
        get_conninfo()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if llm_configured():
        try:
            out = run_citoyen_chat(payload.message, payload.user_id, payload.tenant_id)
        except Exception:
            logger.warning("citoyen agent fallback to template pipeline")
            out = None
        if out is not None:
            rep = out.get("created_report")
            return CitoyenChatOut(
                reply=out["reply"],
                category=str(rep.get("category") or "") if rep else "",
                municipal_service=str(rep.get("municipal_service") or "") if rep else "",
                sentiment_score=float(rep.get("sentiment_score") or 0.0) if rep else 0.0,
                reassured=not bool(rep.get("is_spam")) if rep else True,
            )
    try:
        r = submit_report(payload.user_id, payload.message, tenant_id=payload.tenant_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    cat = r["category"]
    svc = r["municipal_service"]
    sp = bool(r["is_spam"])
    if llm_configured():
        try:
            reply = _citoyen_llm(r, payload.message)
        except Exception:
            logger.warning("citoyen LLM fallback to template")
            reply = _citoyen_template(r)
    else:
        reply = _citoyen_template(r)
    return CitoyenChatOut(
        reply=reply,
        category=cat,
        municipal_service=svc,
        sentiment_score=float(r["sentiment_score"]),
        reassured=not sp,
    )


class MairieQueryIn(BaseModel):
    query: str = Field(..., description="Question en langage naturel (démo)", max_length=5000)


class MairieQueryOut(BaseModel):
    answer: str
    top_reports: list[dict[str, Any]]


@router.post("/chat/mairie", response_model=MairieQueryOut)
@limiter.limit("10/minute")
def chat_mairie(request: Request, payload: MairieQueryIn) -> MairieQueryOut:
    try:
        get_conninfo()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    q = payload.query or ""
    wants = _wants_urgent_top3(q)
    if llm_configured():
        top: list[dict[str, Any]] = []
        if wants:
            top = top_urgent_by_sentiment(days=7, limit=3)
        try:
            answer = _mairie_llm(q, top)
        except Exception:
            logger.warning("mairie LLM fallback to static summary")
            answer = _mairie_fallback(q, top)
        return MairieQueryOut(answer=answer, top_reports=top)
    if not wants:
        return MairieQueryOut(
            answer=(
                "Pour la démo, posez par exemple : « Quels sont les 3 problèmes les plus urgents "
                "basés sur le sentiment des citoyens cette semaine ? »"
            ),
            top_reports=[],
        )
    rows = top_urgent_by_sentiment(days=7, limit=3)
    if not rows:
        return MairieQueryOut(
            answer="Aucun signalement ouvert sur les 7 derniers jours ou base vide.",
            top_reports=[],
        )
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i}. [{r['category']}] score {r['sentiment_score']:.2f} — "
            f"{(r['content'] or '')[:160]}{'…' if len(r.get('content') or '') > 160 else ''}"
        )
    answer = "Les 3 signalements ouverts les plus urgents (sentiment le plus négatif, 7 jours) :\n" + "\n".join(
        lines
    )
    return MairieQueryOut(answer=answer, top_reports=rows)


class AgentChatIn(BaseModel):
    question: str = Field(..., description="Question de l'agent de mairie", max_length=2000)
    tenant_id: str = Field("ia-pipeline", description="Tenant ID", max_length=128)


@router.post("/chat/agent", response_model=AgentChatOut)
@limiter.limit("10/minute")
def chat_agent(request: Request, payload: AgentChatIn) -> AgentChatOut:
    try:
        get_conninfo()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if llm_configured():
        try:
            return run_agent_chat(payload.question)
        except Exception:
            logger.warning("agent endpoint fallback to mairie static mode")
    return mairie_fallback_chat(payload.question)
