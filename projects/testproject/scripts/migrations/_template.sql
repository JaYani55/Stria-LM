-- Migration Template
-- ==================
-- A template for SQL migrations.
--
-- Naming convention: YYYYMMDD_HHMMSS_description.sql
-- Example: 20250101_120000_add_tags_table.sql
--
-- Migrations are applied in order and tracked in schema_versions table.

-- Up Migration
-- Add your schema changes here

-- Example: Create a new table
-- CREATE TABLE IF NOT EXISTS tags (
--     id INTEGER PRIMARY KEY,
--     name TEXT NOT NULL UNIQUE,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- Example: Add a column
-- ALTER TABLE qa_text ADD COLUMN tags TEXT;

-- Example: Create an index
-- CREATE INDEX IF NOT EXISTS idx_qa_text_weight ON qa_text(weight);

-- Down Migration (for rollback - keep commented)
-- DROP TABLE IF EXISTS tags;
-- ALTER TABLE qa_text DROP COLUMN tags;
