from __future__ import annotations

import re
from typing import Any

_SPAM_PATTERNS = re.compile(
    r"(?i)(gagner|gagnez|gratuit|cliquez\s+ici|http[s]?://|"
    r"iphone|bitcoin|viagra|casino|votre\s+compte|suspens|félicitations|"
    r"loterie|héritage|transfert\s+d'argent|crypto)",
)

_FR_NEGATIVE = re.compile(
    r"(?i)(catastrophe|inadmissible|honte|scandale|"
    r"dangereux|urgent|désespér|furieu|exaspér|colère|jamais|[ée]c[œo]ur[ée])"
)
_FR_DISTRESS = re.compile(
    r"(?i)(aidez|aidez-moi|sos|peur|enfant|sécurit[ée].*risque|"
    r"bless[ée]\s*grave|d[ée]truit)",
)
_FR_POSITIVE = re.compile(
    r"(?i)(merci|bravo|bien|satisfait|content|félicit|"
    r"amélior|suggestion|pourrait-on)",
)
_FR_OFFTOPIC = re.compile(
    r"(?i)(recette\s+de|score\s+du\s+match|météo\s+à\slondres)",
)

MIN_CONTENT_LEN = 3


def analyze_spam_sentiment_urgency(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    out: dict[str, Any] = {
        "is_spam": False,
        "spam_reasons": [],
        "sentiment_score": 0.0,
        "urgency": "normal",
        "tone": "neutre",
    }
    if len(t) < MIN_CONTENT_LEN:
        out["is_spam"] = True
        out["spam_reasons"].append("contenu_trop_court_ou_vide")
        return out
    if _SPAM_PATTERNS.search(t):
        out["is_spam"] = True
        out["spam_reasons"].append("mots_clefs_commerciaux_ou_hameçonnage")
    if _FR_OFFTOPIC.search(t) and "mairie" not in t.lower() and "rue" not in t.lower():
        out["is_spam"] = True
        out["spam_reasons"].append("hors_sujet_probable")
    # Score heuristique dans [-1, 1] (le LLM via MCP pourrait affiner côté client)
    score = 0.0
    if _FR_NEGATIVE.search(t):
        score -= 0.45
    if _FR_DISTRESS.search(t):
        score -= 0.35
        out["urgency"] = "haute"
    if _FR_POSITIVE.search(t):
        score += 0.35
    if re.search(r"[!?]{2,}", t):
        score -= 0.15
    if len(t) > 200 and score < 0.2:
        score -= 0.1
    if _FR_DISTRESS.search(t) or (score < -0.3):
        out["urgency"] = "haute"
    elif score < -0.1:
        out["urgency"] = "moyenne"
    if score < -0.4:
        out["tone"] = "colère ou frustration"
    elif _FR_DISTRESS.search(t):
        out["tone"] = "détresse ou insécurité"
    elif score > 0.2:
        out["tone"] = "positif ou suggestion"
    else:
        out["tone"] = "neutre"
    out["sentiment_score"] = max(-1.0, min(1.0, score))
    return out
