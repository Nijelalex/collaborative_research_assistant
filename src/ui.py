# src/ui.py
import streamlit as st
from langgraph_flow import build_graph

st.set_page_config(page_title="Collaborative Research Assistant", page_icon="🧠")
st.title("🧠 Collaborative Research Assistant")

graph = build_graph()
topic = st.text_input("Enter your research topic:")

if st.button("Generate Literature Review") and topic:
    with st.spinner("Collaborating agents working..."):
        state = {"topic": topic}
        result = graph.invoke(state)
    st.markdown("### 🔍 Summary")
    st.write(result["summary"])
    st.markdown("### 🧠 Critique")
    st.write(result["critique"])
    st.markdown("### ✏️ Final Literature Review")
    st.success(result["final"])
