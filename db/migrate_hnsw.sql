-- Migration : ajout de l'index HNSW sur reports.embedding
-- Exécuter : psql -d municipall -f db/migrate_hnsw.sql

CREATE INDEX IF NOT EXISTS idx_reports_embedding_hnsw
  ON reports USING hnsw (embedding vector_cosine_ops);
