CREATE TABLE IF NOT EXISTS rag_store (
    topic TEXT PRIMARY KEY,
    context TEXT,
    summary TEXT,
    critique TEXT,
    citations TEXT,
    final TEXT,
    embedding BLOB
);