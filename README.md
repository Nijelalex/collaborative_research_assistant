# 🧠 AI Research Assistant

A **Generative AI-powered assistant** that helps students and researchers **discover, summarize, and critique academic papers** — to support **assignments, theses, or self-learning**.

This app integrates **retrieval-augmented generation (RAG)** with a multi-agent graph flow, providing a full research workflow — from topic search to literature review generation and Q&A on archived knowledge.

---

## 🚀 Features

- **Semantic Scholar API Integration** — Retrieve recent, relevant research papers based on your topic.  
- **Local Archives (RAG)** — Store and retrieve previously analyzed topics using embeddings.  
- **Document Uploads** — Upload your own **PDF** or **DOCX** research papers for summarization.  
- **Multi-Agent Graph** — workflow created between agents (retrieval, summarization, critique, and final documentation).  
- **Research Memory (SQLite DB)** — Archives topics, embeddings, and feedback for continuous improvement.  
- **Interactive Streamlit UI** — Simple, elegant interface with visualization and history tracking.  
- **LangSmith Integration** — Direct link to your LangSmith dashboard for pipeline monitoring and tracing.  
- **Feedback System (Not working currently)** — Like/dislike feedback for summaries to enhance continuous improvement.  
- **Q&A Module** — Ask follow-up questions based on archived research summaries.

---

## 🧪 Run the App

`streamlit run src/ui.py`

---

## 🦉 Usage Workflow

1. Enter a Topic — e.g. “Large Language Models in Healthcare”.

2. Select Retrieval Mode:

    - Semantic Scholar API — Fetch and analyze live papers.

    - Archives — Search your local RAG database for past analyses.

    - Upload Document — Provide your own PDF/DOCX file for summarization.

3. Generate Literature Review — The app orchestrates multiple AI agents to produce:

    - Summary

    - Critical Analysis

    - Literature Review

    - References

4. Ask Questions — Use the Q&A box to query archived research.

---

## Components

| Component | Description |
|------------|-------------|
| `langgraph_flow.py` | Defines the LangGraph agent flow for research orchestration. |
| `db.py` | Handles SQLite database, embedding search, topic retrieval |
| `ui.py` | The Streamlist UI with all components for seach and qna. |
| `qna_helper.py` | Enables context-based question answering from archives. |
| `images/` | Contains UI assets (logo, icon, graph visualization (persisted)). |
| `static/css/style.css` | Defines UI/UX for Streamlit. |

