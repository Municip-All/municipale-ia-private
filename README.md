# Municip’All — IA, API et reporting

Dépôt **privé** : socle technique pour la plateforme (classification de signalements par **Random Forest**, API **FastAPI**, moteur **reporting** avec **PostgreSQL + pgvector**, outils **MCP** et option **Mistral** pour les réponses conversationnelles).

---

## Prérequis

- **Python 3.10+** (recommandé : 3.12) pour le SDK MCP et les outils récents.
- **PostgreSQL** avec l’extension **pgvector** pour le module reporting (voir `db/schema.sql`).
- **Redis** (optionnel) : cache sur `/predict`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # ou `.venv\Scripts\activate` sous Windows
pip install -r requirements.txt
```

---

## Aperçu du dépôt

| Zone | Rôle principal |
|------|----------------|
| `data_preprocessing.py`, `train_model.py`, `model_inference.py` | Pipeline ML : données → TF-IDF → **Random Forest** → prédiction `/predict`. |
| `api_fastapi.py` | Application FastAPI unifiée : **prédictions** + routes **/reporting/\***. |
| `municipal/` | Logique reporting : embeddings locaux, analyse spam/sentiment, doublons pgvector, routage catégorie/service, client Mistral. |
| `mcp_municipal.py` | Serveur **MCP** (stdio) exposant `smart-analyzer`, `duplicate-finder`, `smart-router`. |
| `db/schema.sql` | Schéma SQL (`reports`, enum de statut, `vector(384)`). |
| `scripts/seed_synthetic_pg.py` | Peuplement synthétique de la base (spams, doublons, sentiments variés). |
| `docs/MCP_INTEGRATION.md` | Configuration MCP, variables Mistral, tests, dépannage. |
| `main.py` | Point d’entrée du pipeline ML complet (prétraitement + entraînement + démo). |
| `qdrant/` | Code lié à Qdrant (expérimentation / autre brique Vector). |

---

## 1. Pipeline ML (Random Forest + API)

Génère les artefacts sous `artifacts/` (CSV, TF-IDF, modèle, métriques) :

```bash
python main.py                    # pipeline complet + démo d’inférence
# ou étape par étape :
python data_preprocessing.py
python train_model.py
```

Lancer l’API :

```bash
uvicorn api_fastapi:app --reload --port 8000
```

**Prédiction (exemple)** :

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description":"Tas de déchets rue Victor Hugo","lat":49.26,"lon":2.44,"hour":10}'
```

Sans Redis, `/predict` fonctionne sans cache.

---

## 2. Reporting (PostgreSQL + signalements)

1. Créer une base et appliquer le schéma :

   ```bash
   psql -d municipall -f db/schema.sql
   ```

2. Exporter la connexion (obligatoire pour insertion et outil duplicate-finder) :

   ```bash
   export DATABASE_URL="postgresql://utilisateur:secret@localhost:5432/municipall"
   ```

3. (Optionnel) Remplir des données de test :

   ```bash
   python scripts/seed_synthetic_pg.py --truncate
   ```

**Routes utiles** (même process `uvicorn` que ci-dessus) :

| Méthode | Route | Description |
|--------|--------|-------------|
| `POST` | `/reporting/submit` | Enregistre un signalement (pipeline analyzer → duplicate → insert). |
| `POST` | `/reporting/chat/citoyen` | Réponse « bot citoyen » (texte fixe ou **Mistral** si clé configurée). |
| `POST` | `/reporting/chat/mairie` | Réponse « dashboard mairie » (Mistral + contexte possible sur les urgences). |

Schéma interactif : une fois l’API lancée, ouvrir **http://127.0.0.1:8000/docs**.

**Mistral (NumSpot)** : définir au minimum `MISTRAL_API_KEY`. Défauts : `MISTRAL_API_BASE=https://api.mistral.numspot.com/v1`, `MISTRAL_MODEL=mistral-medium-2508`. Ne **jamais** commiter la clé : utiliser l’environnement ou un `.env` listé dans `.gitignore`.

Détail des variables et du flux MCP : **`docs/MCP_INTEGRATION.md`**.

---

## 3. Serveur MCP (stdio)

À lancer depuis la racine du projet, avec `DATABASE_URL` si vous utilisez **duplicate-finder** :

```bash
python mcp_municipal.py
```

Intégration **Cursor / Claude Desktop** : voir `docs/MCP_INTEGRATION.md`.

---

## Tests

```bash
pytest tests/test_mistral_client.py -m "not integration"   # hors ligne
pytest tests/test_mistral_client.py -m integration         # appel API Mistral réel si clé présente
```

---

## Auteur et contexte

Mehmet Alkaya — Epitech, spécialisation Data & IA, projet EDP / Municip’All.

Pour la note de synthèse longue sur les choix de modèle et la vision produit, se référer aux livrables pédagogiques du dépôt ou à la documentation projet hors README.
