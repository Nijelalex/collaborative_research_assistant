import streamlit as st
import time
from PIL import Image
import io
from langgraph_flow import build_graph, retriever_node, summarizer_node, critic_node, writer_node

st.set_page_config(page_title="Collaborative Research Assistant", page_icon="🧠", layout="wide")

# Custom CSS for full-width layout and styled boxes
st.markdown(
    """
    <style>
        .block-container {
            max-width: 95% !important;
            padding-top: 2rem;
        }
        .agent-box {
            background-color: #f9f9f9;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .title-box {
            color: #222;
            font-weight: 600;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Collaborative Research Assistant")
st.markdown("Generate research summaries, critiques, and literature reviews using multi-agent collaboration.")

graph = build_graph()

topic = st.text_input("Enter your research topic:", placeholder="e.g. Large Language Models in Healthcare")

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

    # --- Run the full LangGraph pipeline ---
    start_time = time.time()
    state = graph.invoke(state)
    elapsed = time.time() - start_time

    # Final completion message
    progress.progress(1.0)

    if state.get("retrieval_failed", False):
        status.warning("⚠️ Retrieval failed — unable to find relevant papers for your topic.")
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

    png_bytes = graph.get_graph().draw_mermaid_png()

    # convert bytes to PIL image
    image = Image.open(io.BytesIO(png_bytes))

    with st.expander("📊 Show Research Graph (click to expand/close)"):
        st.image(image, caption="Collaborative Research Assistant Graph", use_container_width=True)