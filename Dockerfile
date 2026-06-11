FROM python:3.12-slim

WORKDIR /app

# Arguments de build pour les secrets (optionnels, pour CI/CD)
ARG MISTRAL_API_KEY
ARG MISTRAL_API_BASE
ARG MISTRAL_MODEL
ARG MISTRAL_TIMEOUT_S

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

EXPOSE 8000

# En production, les variables sont injectées via docker-compose ou runtime env
CMD ["python", "-m", "uvicorn", "api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
