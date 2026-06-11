-- Municip'All — schéma PostgreSQL unifié (NestJS backend + IA pipeline)
-- Compatible avec TypeORM (reports.id = INT, tenant_id STRING)
-- Exécuter en superutilisateur sur la base :
--   psql -d municipall -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- type ENUM non utilisé avec NestJS/TypeORM (status libre pour flexibilité)
-- On garde le schéma ouvert pour permettre les valeurs 'En attente', 'En cours', 'Résolu', 'Doublon', 'Rejeté'

-- NOTE: les tables TypeORM hors reports sont gérées par TypeORM synchronize
-- / DatabaseSchemaService.

-- Reports : table MANUELEMENT alignée sur les deux projets
CREATE TABLE IF NOT EXISTS reports (
  id                SERIAL PRIMARY KEY,
  tenant_id         VARCHAR(64) NOT NULL,
  user_id           INTEGER,
  category          VARCHAR(128) NOT NULL,
  status            VARCHAR(64) NOT NULL DEFAULT 'En attente',
  is_resident       BOOLEAN NOT NULL DEFAULT true,
  image_url         TEXT,
  description       TEXT,

  -- champs IA / pipeline enrichissement (ajoutés par le flux backend → IA)
  sentiment_score   REAL,
  ai_confidence     REAL,
  is_spam           BOOLEAN NOT NULL DEFAULT false,
  duplicate_of_id   INTEGER,
  municipal_service VARCHAR(160),
  ai_category       VARCHAR(128),
  ai_processed      BOOLEAN NOT NULL DEFAULT false,

  -- pgvector embedding (384-d via sentence-transformers)
  embedding         vector(384),

  -- PostGIS location (Point, WGS84)
  location geometry(Point, 4326),

  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reports_tenant_id ON reports (tenant_id);
CREATE INDEX IF NOT EXISTS idx_reports_user_id ON reports (user_id);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports (status);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports (created_at);
CREATE INDEX IF NOT EXISTS idx_reports_category ON reports (category);
CREATE INDEX IF NOT EXISTS idx_reports_duplicate_of ON reports (duplicate_of_id);
CREATE INDEX IF NOT EXISTS idx_reports_is_spam ON reports (is_spam);
CREATE INDEX IF NOT EXISTS idx_reports_sentiment ON reports (sentiment_score);

-- Après peuplement : index HNSW sur embedding pour le Duplicate-Finder
-- CREATE INDEX IF NOT EXISTS idx_reports_embedding_hnsw ON reports USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE reports IS 'Signalements citoyens unifiés (NestJS + IA). Embeddings 384D via sentence-transformers pour doublons sémantiques.';
