from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np

from municipal.embeddings import embed_texts

_lock = threading.Lock()
_cached: tuple[np.ndarray, list[dict[str, str]]] | None = None

# Règles + ancres sémantiques (FR) vers catégorie + service
ANCHORS: list[dict[str, str]] = [
    {
        "category": "Voirie",
        "municipal_service": "Services techniques",
        "text": "Nid de poule trou dans la chaussée trottoir cassé voirie signalisation abîmée",
    },
    {
        "category": "Éclairage public",
        "municipal_service": "Services techniques",
        "text": "Lampe cassée panne d'éclairage public obscurité lampadaire allumé toute la nuit",
    },
    {
        "category": "Espaces verts",
        "municipal_service": "Espaces verts",
        "text": "Arbre tombé tonte haie parc banc jardin allée non entretenu",
    },
    {
        "category": "Déchets & propreté",
        "municipal_service": "Propreté",
        "text": "Déchets sauvages encombrants poubelle débordante mégots verre",
    },
    {
        "category": "Urbanisme",
        "municipal_service": "Service urbanisme",
        "text": "Permis de construire recours lotissement hauteur immeuble stationnement PPU",
    },
    {
        "category": "Sécurité & tranquillité",
        "municipal_service": "Police municipale",
        "text": "Tapage agression violence stationnement dangereux feu d'artifice incivilité nuit",
    },
    {
        "category": "Mobilier urbain",
        "municipal_service": "Services techniques",
        "text": "Banc cassé abri bus caddie caddis jeux pour enfants barrière tordue",
    },
    {
        "category": "Eau & assainissement",
        "municipal_service": "Eau & assainissement",
        "text": "Fuite d'eau bouche d'égout inondation regard cassé caniveau",
    },
]


def _get_anchor_matrix() -> tuple[np.ndarray, list[dict[str, str]]]:
    global _cached
    with _lock:
        if _cached is None:
            texts = [a["text"] for a in ANCHORS]
            m = embed_texts(texts, normalize=True)
            _cached = (m, ANCHORS)
        return _cached


def smart_route(text: str) -> dict[str, Any]:
    """Smart-Router : plus proche voisin sur ancres sémantiques (embeddings locaux)."""
    t = (text or "").strip()
    if not t:
        return {
            "category": "Autre",
            "municipal_service": "Secrétariat général",
            "confidence": 0.0,
            "rationale": "Texte vide",
        }
    mat, labels = _get_anchor_matrix()
    v = embed_texts([t], normalize=True)[0]
    sims = mat @ v
    j = int(np.argmax(sims))
    conf = float(sims[j])
    label = labels[j]
    return {
        "category": label["category"],
        "municipal_service": label["municipal_service"],
        "confidence": max(0.0, min(1.0, (conf + 1) / 2)),
        "rationale": f"Rapprochement sémantique avec le thème « {label['category']} » (score {conf:.2f})",
    }


def smart_route_json(text: str) -> str:
    return json.dumps(smart_route(text), ensure_ascii=False)
