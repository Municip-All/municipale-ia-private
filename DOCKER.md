# 🐳 Docker Development Setup

## Configuration des secrets LiteLLM

### 1. Créer ton fichier `.env` local

Copie le fichier exemple :

```bash
cp .env.example .env
```

### 2. Remplir les valeurs dans `.env`

```bash
LITELLM_API_KEY=<votre-clé-injectée-via-opencode>
LITELLM_MODEL=mistral/mistral-medium-2508
LITELLM_API_BASE=
LITELLM_TIMEOUT_S=120
```

### 3. Lancer avec Docker Compose

```bash
# Build et démarrage
docker-compose -f docker-compose.dev.yml up --build

# En arrière-plan
docker-compose -f docker-compose.dev.yml up -d

# Voir les logs
docker-compose -f docker-compose.dev.yml logs -f app
```

### 4. Vérifier que ça fonctionne

```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true}
```

---

## 🔒 Sécurité

### Le fichier `.env` est ignoré par Git

Vérifié dans `.gitignore` :
```
.env
.env.*
!.env.example
```

**Ne jamais commiter `.env` !**

### Utilisation en CI/CD (GitHub Actions)

Les secrets sont injectés automatiquement via les `Actions secrets` :
- `LITELLM_API_KEY`
- `LITELLM_MODEL`
- `LITELLM_API_BASE`
- `LITELLM_TIMEOUT_S`

Voir `.github/workflows/docker-build.yml` pour le workflow complet.

---

## 🎯 Architecture Docker

```
💻 Ton Mac (localhost)
│
├── 🐳 Docker Container: municipall-ia-dev (port 8000)
│   ├── FastAPI app
│   ├── Bert embeddings
│   └── LiteLLM client (via variables d'env)
│
└── 🗄️ PostgreSQL (localhost:5432, hors Docker)
    └── pgvector extension
```

---

## 📋 Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `LITELLM_API_KEY` | Clé API LLM (injectée via OpenCode) | ✅ |
| `LITELLM_MODEL` | Nom du modèle (prefix LiteLLM) | ✅ |
| `LITELLM_API_BASE` | URL de base de l'API (optionnel) | ❌ |
| `LITELLM_TIMEOUT_S` | Timeout requêtes (s) | ✅ |
| `DATABASE_HOST` | Hôte PostgreSQL | ✅ |
| `DATABASE_PORT` | Port PostgreSQL | ✅ |
| `DATABASE_USER` | Utilisateur PostgreSQL | ✅ |
| `DATABASE_PASSWORD` | Mot de passe PostgreSQL | ✅ |
| `DATABASE_NAME` | Nom de la base | ✅ |
| `IA_PORT` | Port exposé (défaut: 8000) | ❌ |
