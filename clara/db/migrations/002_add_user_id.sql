-- Migration 002: Add user_id tenant partition column
-- This column is nullable for backwards-compatibility with pre-tenant data.
-- New records should always set user_id explicitly.

ALTER TABLE memories ADD COLUMN user_id TEXT;

CREATE INDEX ix_memories_user_id ON memories(user_id);
CREATE INDEX ix_memories_user_type_status ON memories(user_id, memory_type, status);
