-- Run once against your EXISTING PostgreSQL database before using stale-job recovery.
-- This preserves all existing rows and data.
ALTER TABLE submissions
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- Backfill is automatic for existing rows because of the DEFAULT above.
