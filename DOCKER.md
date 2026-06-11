# 🐳 Docker Development Setup

## Configuration des secrets Mistral

### 1. Créer ton fichier `.env` local

Copie le fichier exemple :

```bash
cp .env.example .env
```

### 2. Remplir les valeurs dans `.env`

```bash
MISTRAL_API_KEY=HXjrM2J8aShhHwIhRv0xFFoNd957TONB
MISTRAL_API_BASE=https://api.mistral.numspot.com/v1
MISTRAL_MODEL=mistral-medium-2508
MISTRAL_TIMEOUT_S=120
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
- `MISTRAL_API_KEY`
- `MISTRAL_API_BASE`
- `MISTRAL_MODEL`
- `MISTRAL_TIMEOUT_S`

Voir `.github/workflows/docker-build.yml` pour le workflow complet.

---

## 🎯 Architecture Docker

```
💻 Ton Mac (localhost)
│
├── 🐳 Docker Container: municipall-ia-dev (port 8000)
│   ├── FastAPI app
│   ├── Bert embeddings
│   └── Mistral client (via variables d'env)
│
└── 🗄️ PostgreSQL (localhost:5432, hors Docker)
    └── pgvector extension
```

---

## 📋 Variables d'environnement

| Variable | Description | Obligatoire |
|----------|-------------|-------------|
| `MISTRAL_API_KEY` | Clé API Mistral NumSpot | ✅ |
| `MISTRAL_API_BASE` | URL de base de l'API | ✅ |
| `MISTRAL_MODEL` | Nom du modèle | ✅ |
| `MISTRAL_TIMEOUT_S` | Timeout requêtes (s) | ✅ |
| `DATABASE_HOST` | Hôte PostgreSQL | ✅ |
| `DATABASE_PORT` | Port PostgreSQL | ✅ |
| `DATABASE_USER` | Utilisateur PostgreSQL | ✅ |
| `DATABASE_PASSWORD` | Mot de passe PostgreSQL | ✅ |
| `DATABASE_NAME` | Nom de la base | ✅ |
| `IA_PORT` | Port exposé (défaut: 8000) | ❌ |
