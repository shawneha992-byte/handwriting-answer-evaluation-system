-- Run once on your EXISTING PostgreSQL database. Each exam now belongs to the teacher who created it.
ALTER TABLE exams ADD COLUMN IF NOT EXISTS owner_id INTEGER REFERENCES users(id);
CREATE INDEX IF NOT EXISTS ix_exams_owner_id ON exams (owner_id);

-- Exams created before this change have no owner, so no teacher can see them. Give them to the right account.
-- This line hands ALL of them to the first registered user; edit the WHERE/SET if you have several teachers.
UPDATE exams SET owner_id = (SELECT id FROM users ORDER BY id LIMIT 1) WHERE owner_id IS NULL;
