# src/ui.py
import streamlit as st
import time
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

    state = {"topic": topic}

    # Manual execution for real-time feedback
    steps = [
    (retriever_node, "🔍 Retrieving top papers from Semantic Scholar..."),
    (summarizer_node, "🧩 Summarizing research findings..."),
    (critic_node, "🧠 Critiquing methodology and identifying gaps..."),
    (writer_node, "✏️ Writing literature review..."),
    ]

    for i, (node_func, message) in enumerate(steps):
        status.info(message)
        state = node_func(state)
        progress.progress((i + 1) / len(steps))
        time.sleep(0.8)
    
    # Extract outputs
    summary = state["summary"].content
    critique = state["critique"].content
    review = state["final"].content
    citations = state.get("citations", "No references available.")

    # Final completion message
    progress.progress(1.0)
    status.success("✅ All agents completed successfully!")

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
