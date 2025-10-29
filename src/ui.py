import streamlit as st
from langgraph_flow import build_graph

st.set_page_config(page_title="Collaborative Research Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Collaborative Research Assistant")

graph = build_graph()
topic = st.text_input("Enter your research topic:")

if st.button("Generate Literature Review") and topic:
    # Prepare columns for visual layout
    col_status, col_spacer = st.columns([1, 3])
    with col_status:
        status_box = st.empty()

    with st.spinner("Collaborating agents are working..."):
        # Initialize state
        state = {"topic": topic}
        status_box.info("🔍 Retrieving top papers from Semantic Scholar...")
        result = graph.invoke(state)

    # Extract outputs
    summary = result["summary"].content
    critique = result["critique"].content
    review = result["final"].content
    citations = result.get("citations","No citations available")
        
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


    st.success("✅ Research summary complete!")