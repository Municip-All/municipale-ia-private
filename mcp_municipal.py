from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from municipal.analyzer import smart_analyzer_json
from municipal.duplicate import duplicate_finder_json
from municipal.router import smart_route_json

mcp = FastMCP("Municip'All / Reporting")


@mcp.tool(name="smart-analyzer")
def smart_analyzer(content: str, user_id: str = "") -> str:
    """
    A — Validation en temps réel d'un signalement : spam, sentiment (-1..1), ton, urgence,
    embedding 384D (génération locale, hors envoi de texte vers un service tiers si modèle local).
    """
    return smart_analyzer_json(content, user_id if user_id else None)


@mcp.tool(name="duplicate-finder")
def duplicate_finder_tool(
    embedding: list[float],
    exclude_report_id: str = "",
    threshold: float = 0.85,
) -> str:
    """
    B — Recherche du signalement le plus proche (similarité cosinus > seuil) dans PostgreSQL+pgvector.
    exclude_report_id : ID (entier) du signalement courant à exclure (ex. déjà inséré) si besoin.
    """
    ex = int(exclude_report_id.strip()) if exclude_report_id.strip() else None
    return duplicate_finder_json(embedding, ex, threshold)


@mcp.tool(name="smart-router")
def smart_router(content: str) -> str:
    """
    C — Routage sémantique (catégorie métier ex. Voirie, et service municipal cible) via ancres
    d'embedding.
    """
    return smart_route_json(content)


if __name__ == "__main__":
    mcp.run()
