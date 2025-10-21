CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS face_embeddings CASCADE;
DROP TABLE IF EXISTS persons CASCADE;

CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    first_name TEXT NULL DEFAULT NULL,
    last_name TEXT NULL DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
    embedding VECTOR(512),
    source_image TEXT,
    image_hash TEXT,
    model TEXT DEFAULT 'insightface_buffalo_l_v1',
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optional index for fast nearest neighbor searches
CREATE INDEX face_embeddings_embedding_idx
    ON face_embeddings
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

-- Easier to query for file path/image hash
CREATE INDEX idx_face_embeddings_source_image ON face_embeddings(source_image);
CREATE INDEX idx_face_embeddings_image_hash ON face_embeddings(image_hash);