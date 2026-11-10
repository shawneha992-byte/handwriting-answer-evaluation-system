-- Run once on an existing PostgreSQL database. The new knowledge_docs / knowledge_chunks tables are created
-- automatically on startup; these statements add the columns to tables that already exist.
ALTER TABLE questions ADD COLUMN IF NOT EXISTS rubric_json TEXT;
ALTER TABLE questions ADD COLUMN IF NOT EXISTS model_answer TEXT DEFAULT '';
ALTER TABLE answers ALTER COLUMN student_answer TYPE TEXT;
ALTER TABLE answers ADD COLUMN IF NOT EXISTS feedback TEXT;
ALTER TABLE answers ADD COLUMN IF NOT EXISTS citations_json TEXT;
ALTER TABLE answers ADD COLUMN IF NOT EXISTS retrieval_score DOUBLE PRECISION;
ALTER TABLE answers ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;
