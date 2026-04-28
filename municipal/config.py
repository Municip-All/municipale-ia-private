import os

EMBEDDING_MODEL = os.environ.get(
    "MUNICIPAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
# paraphrase-multilingual-MiniLM-L12-v2 : 384 dims, multilingue FR
# Alternative : all-MiniLM-L6-v2 (anglais) — fixer MUNICIPAL_EMBEDDING_MODEL si besoin

DUPLICATE_SIMILARITY_THRESHOLD = float(
    os.environ.get("MUNICIPAL_DUPLICATE_THRESHOLD", "0.85")
)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Mistral (NumSpot) — clé jamais committée ; définir MISTRAL_API_KEY dans l'environnement
MISTRAL_API_BASE = os.environ.get(
    "MISTRAL_API_BASE", "https://api.mistral.numspot.com/v1"
).rstrip("/")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-medium-2508")
MISTRAL_TIMEOUT_S = float(os.environ.get("MISTRAL_TIMEOUT_S", "120"))
