# Municip’All — Outils MCP (Reporting)

Ce document décrit comment lancer le serveur **Model Context Protocol (MCP)** et quels noms d’outils le client (Cursor, Claude Desktop, ou orchestrateur interposant un LLM) doit invoquer.

## Prérequis

- **Python 3.10+** (le SDK `mcp` n’est pas installable sous Python 3.9).
- **PostgreSQL** avec l’extension **pgvector** (voir `db/schema.sql`).
- Variable **`DATABASE_URL`**, par exemple :  
  `postgresql://user:mdp@localhost:5432/municipall`  
- Embeddings générés **localement** par défaut via `sentence-transformers` (aucun envoi de texte vers un fournisseur d’API pour les vecteurs, sauf si vous remplacez le module d’embedding).

## Installation (environnement virtuel recommandé)

```bash
cd municipale-ia-private
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Initialiser la base :

```bash
psql -d municipall -f db/schema.sql
export DATABASE_URL=postgresql://user:mdp@localhost:5432/municipall
python scripts/seed_synthetic_pg.py --truncate
```

## Démarrage du serveur MCP (stdio)

Le point d’entrée est `mcp_municipal.py`. Depuis le répertoire du projet, avec `PYTHONPATH` correct (répertoire courant) et `DATABASE_URL` définie :

```bash
export DATABASE_URL=postgresql://user:mdp@localhost:5432/municipall
cd /chemin/vers/municipale-ia-private
source .venv/bin/activate
python mcp_municipal.py
```

Le serveur écoute sur **stdin/stdout** (transport MCP standard), sans port réseau.

## Configuration dans Cursor (exemple `mcp.json`)

```json
{
  "mcpServers": {
    "municipall-reporting": {
      "command": "/absolu/vers/municipale-ia-private/.venv/bin/python",
      "args": ["/absolu/vers/municipale-ia-private/mcp_municipal.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:mdp@localhost:5432/municipall",
        "MUNICIPAL_EMBEDDING_MODEL": "paraphrase-multilingual-MiniLM-L12-v2"
      }
    }
  }
}
```

- **`MUNICIPAL_EMBEDDING_MODEL`** : modèle Sentence-Transformers (sortie **384** dimensions, aligné sur `VECTOR(384)`).  
  Vous pouvez utiliser `all-MiniLM-L6-v2` si vous acceptez un biais anglophone ; le modèle multilingue est en général plus adapté au français.

- **`MUNICIPAL_DUPLICATE_THRESHOLD`** : seuil de similarité cosinus (défaut `0.85`) pour l’outil `duplicate-finder`.

## Outils exposés (noms d’appel)

| Nom de l’outil     | Rôle |
|--------------------|------|
| `smart-analyzer`   | Filtre spam, sentiment/urgence/ton, génère l’**embedding** (local). |
| `duplicate-finder` | Recherche en base le signalement le plus proche (pgvector) ; similarité supérieure au seuil configuré → doublon. |
| `smart-router`   | Propose **catégorie** (ex. Voirie) et **service municipal** cible. |

Contenu des réponses : chaînes **JSON** (le LLM lit la structure : scores, champs, etc.).

Chaînage côté client recommandé pour un signalement :  
`smart-analyzer` → `smart-router` (si le texte n’est pas spam) → `duplicate-finder` (avec l’embedding retourné) → écriture SQL applicative.  
L’API FastAPI **`POST /reporting/submit`** exécite déjà ce pipeline (`municipal/pipeline.py`).

## Intégration LLM (Claude, GPT, etc.)

1. L’hôte branche le transport **stdio** vers `python mcp_municipal.py`.  
2. L’hôte interroge `tools/list` (géré par le SDK) et mappe les noms ci-dessus.  
3. Pour chaque outil, les arguments passent tels quels (ex. `content`, `user_id` pour `smart-analyzer`).  
4. Les sorties textuelles JSON peuvent être parsées par le modèle pour décider d’enregistrer, marquer en **Spam** / **Duplicate**, etc.

Aucun secret ne doit figurer en clair dans le code : utilisez l’environnement (`DATABASE_URL`) et, en production, un coffre (Vault, paramètres hébergés, etc.).

## Sécurité des requêtes SQL

Le module `municipal/db.py` n’insère le texte utilisateur qu’en **paramètres** (`%s` / `psycopg`), y compris les vecteurs, sans concaténation de chaînes dans la requête. Cela limite l’injection SQL.

## API de simulation (chatbots)

Avec l’API FastAPI (`uvicorn api_fastapi:app --reload`) :

- `POST /reporting/chat/citoyen` : message rassurant + thématique, après enregistrement.  
- `POST /reporting/chat/mairie` : démo « dashboard textuel » — si la requête évoque urgence / sentiment / « cette semaine », renvoie les 3 signalements **Open** les plus urgents (sentiment le plus négatif) sur 7 jours.  
- `POST /reporting/submit` : enregistrement direct du pipeline complet.

## LLM — Mistral (NumSpot) pour les chats API

Les routes `POST /reporting/chat/citoyen` et `POST /reporting/chat/mairie` appellent l’API **chat completions** (format compatible OpenAI) si une clé est fournie. Les embarquements restent **locaux** ; seule la **réponse conversationnelle** transite par Mistral.

Variables d’environnement (la clé n’est jamais à versionner) :

| Variable | Exemple / défaut |
|----------|------------------|
| `MISTRAL_API_KEY` | *(à remplir)* |
| `MISTRAL_API_BASE` | `https://api.mistral.numspot.com/v1` |
| `MISTRAL_MODEL` | `mistral-medium-2508` |
| `MISTRAL_TIMEOUT_S` | `120` |

Sans `MISTRAL_API_KEY`, le bot **citoyen** utilise le texte prédéfini ; le bot **mairie** conserve l’ancienne démo par mots-clés.

### Dépannage

- **401 Unauthorized** : la passerelle refuse `Authorization: Bearer …`. Causes fréquentes : clé expirée ou révoquée, besoin d’un flux NumSpot (jeton IAM à obtenir séparément), ou mauvaise configuration côté console NumSpot.

### Tests

```bash
# Unités (sans réseau)
pytest tests/test_mistral_client.py -m "not integration"

# Intégration (clé uniquement en variable d’environnement)
MISTRAL_API_KEY="…" MISTRAL_API_BASE="https://api.mistral.numspot.com/v1" MISTRAL_MODEL="mistral-medium-2508" \
  pytest tests/test_mistral_client.py -m integration
```

Code : `municipal/mistral_client.py` et `municipal/config.py`.

Document généré pour le lot Reporting / MCP — Municip’All.
