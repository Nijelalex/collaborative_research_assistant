import streamlit as st
from pathlib import Path
import time
import base64
from langgraph_flow import build_graph
from db import init_db, save_rag_entry_with_embedding, search_similar_rag

st.set_page_config(page_title="Collaborative Research Assistant", page_icon="🧠", layout="wide")


# --- Base directory of project ---
BASE_DIR = Path.cwd()
GRAPH_PATH = Path.cwd() / "images" / "graph.png"

with open(GRAPH_PATH, "rb") as f:
    graph_image_base64 = base64.b64encode(f.read()).decode("utf-8")

# --- CSS & JS paths ---
css_path = BASE_DIR / "static" / "css" / "style.css"
js_path  = BASE_DIR / "static" / "js" / "graph_modal.js"

# Load external CSS & JS
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open(js_path) as f:
    st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)

st.markdown(f"""<div class="right-bar">
    <p>📊 Graph Overview</p>
    <img src="data:image/png;base64,{graph_image_base64}" style="max-width: 100%; height: auto; margin-bottom: 20px;">
</div>
""", unsafe_allow_html=True)

st.title("🧠 Collaborative Research Assistant")
st.markdown("Generate research summaries, critiques, and literature reviews using multi-agent collaboration.")

graph = build_graph()

topic = st.text_input("Enter your research topic:", placeholder="e.g. Large Language Models in Healthcare")

method = st.radio("Choose retrieval method:", ["Semantic API", "Archives"])


if st.button("Generate Literature Review") and topic:
    st.info(f"🚀 Starting research process for: **{topic}**")

    # Create progress bar and status placeholder
    progress = st.progress(0)
    status = st.empty()

    state = {
        "student_id": "S001", # Dummy (until user session created)
        "topic": topic,
        "context": "",
        "summary": "",
        "critique": "",
        "citations": "",
        "final": "", 
        "retrieval_failed": False
        }

    # --- Integrate RAG / Archives ✅
    start_time = time.time()

    if method == "Archives":
        # Starting DB
        init_db()
        # Try semantic similarity search in RAG
        rag_results = search_similar_rag(topic, top_k=1)
        if rag_results:
            st.success(f"✅ Found similar topic in Archives: {rag_results[0][1]['topic']}")
            state.update(rag_results[0][1])
        else:
            st.warning("⚠️ No similar topic found in Archives, fetching via Semantic API...")
            state = graph.invoke(state)
            if not state.get("retrieval_failed", False):
                save_rag_entry_with_embedding(state)
    else:
        # Semantic API selected
        state = graph.invoke(state)
        if not state.get("retrieval_failed", False):
            save_rag_entry_with_embedding(state)

    elapsed = time.time() - start_time


    # Final completion message
    progress.progress(1.0)

    if state.get("retrieval_failed", False):
        status.warning("⚠️ Retrieval failed — Semantic API request free tier exceeded.")
    else:
        status.success(f"✅ All agents completed in {elapsed:.1f} seconds!")
    
    # ---- safe extraction ----
    def safe_get(value, default="Not available"):
        if not value:
            return default
        return getattr(value, "content", value)
    
    # Extract outputs
    summary = safe_get(state.get("summary"),default="Semantic API free requests exceeded")
    critique = safe_get(state.get("critique"), default="🧐 Critique unavailable")
    review = safe_get(state.get("final"),default="📚 Literature review unavailable")
    citations = safe_get(state.get("citations"),default="🔖 References pending")

    # Display results in 2x2 grid layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="title-box">🔍 Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="agent-box">{summary}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="title-box">🧠 Critique</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="agent-box">{critique}</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="title-box">✏️ Literature Review</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="agent-box">{review}</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="title-box">📚 References</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="agent-box">{citations}</div>', unsafe_allow_html=True)

    st.success("🎉 Research summary complete!")