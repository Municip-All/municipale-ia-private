-- Municip'All — schéma PostgreSQL + pgvector
-- Exécuter en superutilisateur sur une base vierge ou après création de la base :
--   createdb municipall
--   psql -d municipall -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE report_status AS ENUM (
  'Open',
  'In_Progress',
  'Resolved',
  'Duplicate',
  'Spam'
);

CREATE TABLE reports (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL,
  content       TEXT NOT NULL,
  category      VARCHAR(128) NOT NULL,
  status        report_status NOT NULL DEFAULT 'Open',
  sentiment_score REAL NOT NULL CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
  embedding     vector(384) NOT NULL,
  duplicate_of_id UUID REFERENCES reports (id) ON DELETE SET NULL,
  municipal_service VARCHAR(160),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reports_user_id ON reports (user_id);
CREATE INDEX idx_reports_status ON reports (status);
CREATE INDEX idx_reports_created_at ON reports (created_at);
CREATE INDEX idx_reports_category ON reports (category);
CREATE INDEX idx_reports_duplicate_of ON reports (duplicate_of_id);

-- Après peuplement : index HNSW ou IVFFlat sur embedding pour accélérer le Duplicate-Finder, ex. :
-- CREATE INDEX idx_reports_embedding_hnsw ON reports USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE reports IS 'Signalements citoyens : embeddings 384D (ex. all-MiniLM-L6-v2) pour doublons sémantiques.';
