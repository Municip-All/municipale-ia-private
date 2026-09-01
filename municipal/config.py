import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

EMBEDDING_MODEL = os.environ.get(
    "MUNICIPAL_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

DUPLICATE_SIMILARITY_THRESHOLD = float(
    os.environ.get("MUNICIPAL_DUPLICATE_THRESHOLD", "0.85")
)

ROUTER_MIN_COSINE_SIM = float(os.environ.get("MUNICIPAL_ROUTE_MIN_COSINE", "0.22"))

def _build_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    host = os.environ.get("DATABASE_HOST", "localhost")
    port = os.environ.get("DATABASE_PORT", "5432")
    user = os.environ.get("DATABASE_USER", "postgres")
    password = os.environ.get("DATABASE_PASSWORD", "")
    db = os.environ.get("DATABASE_NAME", "municipall")
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return f"postgresql://{user}@{host}:{port}/{db}"

_database_url = None

def get_database_url() -> str:
    global _database_url
    if _database_url is None:
        _database_url = _build_database_url()
    return _database_url

LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "openai/glm-53-flash")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "").strip()
LITELLM_API_BASE = os.environ.get("LITELLM_API_BASE", "").strip()
LITELLM_TIMEOUT_S = float(os.environ.get("LITELLM_TIMEOUT_S", "120"))
