import os

EMBEDDING_MODEL = os.environ.get(
    "MUNICIPAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
# paraphrase-multilingual-MiniLM-L12-v2 : 384 dims, multilingue FR
# Alternative : all-MiniLM-L6-v2 (anglais) — fixer MUNICIPAL_EMBEDDING_MODEL si besoin

DUPLICATE_SIMILARITY_THRESHOLD = float(
    os.environ.get("MUNICIPAL_DUPLICATE_THRESHOLD", "0.85")
)

# Cosinus max(ancrage) sous ce seuil → catégorie « Autre » (bruit / faible confiance)
ROUTER_MIN_COSINE_SIM = float(os.environ.get("MUNICIPAL_ROUTE_MIN_COSINE", "0.22"))

def _build_database_url() -> str:
    """Construit DATABASE_URL depuis l'URL complète ou les variables séparées."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    # Fallback : variables séparées (compatibilité NestJS)
    host = os.environ.get("DATABASE_HOST", "localhost")
    port = os.environ.get("DATABASE_PORT", "5432")
    user = os.environ.get("DATABASE_USER", "postgres")
    password = os.environ.get("DATABASE_PASSWORD", "")
    db = os.environ.get("DATABASE_NAME", "municipall")
    if not password:
        raise RuntimeError(
            "DATABASE_PASSWORD is required when not using DATABASE_URL. "
            "Set DATABASE_PASSWORD in your environment."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

DATABASE_URL = _build_database_url()

# LiteLLM — proxy universel (Mistral, OpenAI, Anthropic…)
# Définir LITELLM_API_KEY dans l'environnement (injecté via OpenCode)
LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "mistral/mistral-medium-2508")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "").strip()
LITELLM_API_BASE = os.environ.get("LITELLM_API_BASE", "").strip()
LITELLM_TIMEOUT_S = float(os.environ.get("LITELLM_TIMEOUT_S", "120"))
