FROM python:3.12-slim

WORKDIR /app

# Arguments de build pour les secrets (optionnels, pour CI/CD)
ARG LITELLM_API_KEY
ARG LITELLM_API_BASE
ARG LITELLM_MODEL
ARG LITELLM_TIMEOUT_S

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MUNICIPAL_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MUNICIPAL_EMBEDDING_MODEL}')"

COPY . .

EXPOSE 8000

# En production, les variables sont injectées via docker-compose ou runtime env
CMD ["python", "-m", "uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
