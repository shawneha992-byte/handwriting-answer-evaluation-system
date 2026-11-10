-- PostgreSQL migration: support long answer keys and richer question type labels.
ALTER TABLE questions ALTER COLUMN correct_answer TYPE TEXT;
ALTER TABLE answers ALTER COLUMN correct_answer TYPE TEXT;
ALTER TABLE questions ALTER COLUMN type TYPE VARCHAR(30);
