from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sqlite3
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

DB_PATH = "outputs/rag_store.db"

def init_db():
    import sqlite3
    DB_PATH = "outputs/rag_store.db"
    sql = """
    CREATE TABLE IF NOT EXISTS rag_store (
        topic TEXT PRIMARY KEY,
        context TEXT,
        summary TEXT,
        critique TEXT,
        citations TEXT,
        final TEXT,
        embedding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(sql)
    conn.commit()
    conn.close()

model = get_embedding_model()

def save_rag_entry_with_embedding(state, DB_PATH="outputs/rag_store.db"):
    if state.get("retrieval_failed"):
        return

    # Combine context + summary for embedding
    text_to_embed = state.get("context","") + " " + state.get("summary","").content
    embedding_vector = model.encode(text_to_embed, normalize_embeddings=True)
    embedding_blob = pickle.dumps(embedding_vector)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO rag_store (topic, context, summary, critique, citations, final, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        state["topic"],
        state.get("context",""),
        state.get("summary","").content,
        state.get("critique","").content,
        state.get("citations",""),
        state.get("final","").content,
        embedding_blob
    ))
    conn.commit()
    conn.close()

def search_similar_rag(query, top_k=3, DB_PATH="outputs/rag_store.db"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT topic, context, summary, critique, citations, final, embedding FROM rag_store")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    # Compute query embedding
    query_vec = model.encode(query, normalize_embeddings=True)

    results = []
    for row in rows:
        embedding_vec = pickle.loads(row[6])
        sim = cosine_similarity([query_vec], [embedding_vec])[0][0]
        results.append((sim, {
            "topic": row[0],
            "context": row[1],
            "summary": row[2],
            "critique": row[3],
            "citations": row[4],
            "final": row[5]
        }))

    # Sort by similarity descending
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]

def get_recent_topics(limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT topic, created_at FROM rag_store ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    # Convert to list of dicts for Streamlit table
    return [{"topic": r[0], "date": r[1]} for r in rows]

def save_feedback(topic, content, feedback):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO rag_feedback (topic, content, feedback, created_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    """, (topic, content, feedback))
    conn.commit()
    conn.close()