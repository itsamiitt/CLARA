-- ============================================================================
-- CLARA — Unified Memory Store: Initial Migration
-- 001_init.sql
--
-- Creates the memories table, enums, and indexes for the Unified Memory Store.
-- Requires: PostgreSQL 15+ with pgvector extension.
-- ============================================================================

-- Enable pgvector if not already loaded
CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- Enum types
-- ----------------------------------------------------------------------------

CREATE TYPE memory_type_enum AS ENUM (
    'belief',
    'event',
    'skill',
    'world_model'
);

CREATE TYPE memory_status_enum AS ENUM (
    'active',
    'superseded',
    'deprecated',
    'archived'
);

-- ----------------------------------------------------------------------------
-- Unified memories table
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memories (
    memory_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type     memory_type_enum    NOT NULL,
    content         JSONB               NOT NULL,
    embedding       vector(1536),
    confidence      FLOAT               NOT NULL DEFAULT 0.5,
    status          memory_status_enum  NOT NULL DEFAULT 'active',
    decay_rate      FLOAT               NOT NULL DEFAULT 0.02,
    created_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT now(),
    metadata        JSONB               DEFAULT '{}'::jsonb
);

-- ----------------------------------------------------------------------------
-- Indexes
-- ----------------------------------------------------------------------------

-- Single-column indexes for common filter predicates
CREATE INDEX ix_memories_memory_type ON memories (memory_type);
CREATE INDEX ix_memories_status      ON memories (status);
CREATE INDEX ix_memories_created_at  ON memories (created_at);

-- Composite index for the most common query pattern:
-- "give me all active beliefs", "give me all active skills", etc.
CREATE INDEX ix_memories_type_status ON memories (memory_type, status);

-- HNSW vector similarity index (cosine distance) for semantic search.
-- Tune m and ef_construction based on dataset size.
CREATE INDEX ix_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index on content JSONB for fast key-path lookups
-- (e.g. content->>'subject' = 'user')
CREATE INDEX ix_memories_content ON memories USING gin (content);

-- GIN index on metadata JSONB for auxiliary queries
CREATE INDEX ix_memories_metadata ON memories USING gin (metadata);

-- ----------------------------------------------------------------------------
-- Auto-update trigger for updated_at
-- ----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
