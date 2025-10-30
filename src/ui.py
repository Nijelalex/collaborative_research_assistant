import streamlit as st
from pathlib import Path
import time
import base64
import pandas as pd
from langgraph_flow import build_graph
from db import (
    init_db, 
    save_rag_entry_with_embedding, 
    search_similar_rag, 
    get_recent_topics, 
    save_feedback
)

st.set_page_config(page_title="Collaborative Research Assistant", page_icon="🧠", layout="wide")


# --- Base directory of project ---
BASE_DIR = Path.cwd()
GRAPH_PATH = Path.cwd() / "images" / "graph.png"
LOGO_PATH = Path.cwd() / "images" / "logo.png"

with open(GRAPH_PATH, "rb") as f:
    graph_image_base64 = base64.b64encode(f.read()).decode("utf-8")

with open(LOGO_PATH, "rb") as f:
    logo_image_base64 = base64.b64encode(f.read()).decode("utf-8")


# --- CSS & JS paths ---
css_path = BASE_DIR / "static" / "css" / "style.css"

# Load external CSS & JS
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(f"""<div class="left-bar">
    <img src="data:image/png;base64,{logo_image_base64}" style="max-width: 100%; height: 140px; margin-bottom: 20px;">
    <a href="https://smith.langchain.com/o/161479c6-ccc7-4a79-ab5b-8142f6f7ffa0/projects/p/3b9b18d9-9cf6-44be-86d7-900229f33406?timeModel=%7B%22duration%22%3A%227d%22%7D">Open LangSmith Dashboard ↗️</a>    
    <div class="graph_bar">
    <p id="graph_overview">Graph Overview 📊</p>
    <img src="data:image/png;base64,{graph_image_base64}" style="max-width: 100%; height: 410px; margin: auto; display: block">
    </div>
    </div>
""", unsafe_allow_html=True)

st.title("🧠 Collaborative Research Assistant")
st.markdown("Generate research summaries, critiques, and literature reviews using multi-agent collaboration.")

graph = build_graph()

topic = st.text_input("Enter your research topic:", placeholder="e.g. Large Language Models in Healthcare")

# Starting DB
init_db()

# Get table of topics asked before
with st.expander("📜 Recently Asked Topics", expanded=False):
# Fetch recent topics from DB
    recent_topics = get_recent_topics(limit=10)  # returns a list of dicts

    if recent_topics:
        df = pd.DataFrame(recent_topics)  # Convert to DataFrame for nice table display
        st.table(df)  # Streamlit's built-in table
    else:
        st.info("No recent topics found.")



method = st.radio("Choose retrieval method:", ["Semantic API", "Archives"])


if st.button("Generate Literature Review") and topic:
    st.info(f"🚀 Starting research process for: **{topic}**")
    # Create progress bar and status placeholder
    progress = st.progress(0)
    status = st.empty()

    
    def status_update(msg):
        status.info(msg)

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
    
        # Try semantic similarity search in RAG
        rag_results = search_similar_rag(topic, top_k=1)
        if rag_results:
            st.success(f"✅ Found similar topic in Archives: {rag_results[0][1]['topic']}")
            state.update(rag_results[0][1])
        else:
            st.warning("⚠️ No similar topic found in Archives, fetching via Semantic API...")
            state = graph.invoke(state, callback=status_update)
            if not state.get("retrieval_failed", False):
                save_rag_entry_with_embedding(state)
    else:
        # Semantic API selected
        state = graph.invoke(state, callback=status_update)
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

        # --- Initialize session state ---
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "feedback_type" not in st.session_state:
        st.session_state.feedback_type = None

    # Make buttons close together
    _, feedback_col1, feedback_col2, = st.columns([2.8, 0.15, 0.15])

    # Show buttons only if summary is available
    if summary and not st.session_state.feedback_given:
        with feedback_col1:
            if st.button("👍", key="like_btn"):
                st.session_state.feedback_given = True
                st.session_state.feedback_type = "like"

        with feedback_col2:
            if st.button("👎", key="dislike_btn"):
                st.session_state.feedback_given = True
                st.session_state.feedback_type = "dislike"

    # Handle feedback after click (on next rerun)
    if st.session_state.feedback_given:
        feedback_type = st.session_state.feedback_type
        save_feedback(topic, summary, feedback_type)

        if feedback_type == "like":
            st.success("✅ Thanks for your 👍 Like feedback!")
        else:
            st.warning("⚠️ Got it — your 👎 Dislike has been recorded.")


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