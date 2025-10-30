# src/langgraph_flow.py
from langgraph.graph import StateGraph, END
from agents.retriever_agent import get_retriever_agent
from agents.summarizer_agent import get_summarizer_agent
from agents.critic_agent import get_critic_agent
from agents.writer_agent import get_writer_agent
from typing import TypedDict
import logging

logging.basicConfig(level=logging.INFO)

class ResearchAssistantState(TypedDict):
    """State for the research assistant workflow"""
    # Customer info
    student_id: str
    topic: str
    context: str
    summary: str
    critique: str
    citations: str
    final: str
    retrieval_failed: bool

def retriever_node(state: ResearchAssistantState) -> ResearchAssistantState:
    retrieve = get_retriever_agent()
    state["retrieval_failed"], state["citations"], state["context"] = retrieve(state["topic"])
    return state

def summarizer_node(state: ResearchAssistantState) -> ResearchAssistantState:
    summarize = get_summarizer_agent()
    state["summary"] = summarize(state["context"])
    return state

def critic_node(state: ResearchAssistantState) -> ResearchAssistantState:
    critique = get_critic_agent()
    state["critique"] = critique(state["summary"])
    return state

def writer_node(state: ResearchAssistantState) -> ResearchAssistantState:
    writer = get_writer_agent()
    state["final"] = writer(state["summary"], state["critique"])
    return state

def build_graph():
    g = StateGraph(ResearchAssistantState)
    g.add_node("retriever", retriever_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

   
    def check_retrieval(state: ResearchAssistantState) -> str:        
        # return "END" if state.get("retrieval_failed") else "summarizer"
        logging.info(f"my_router_function called with state: {state}")
        if state.get("retrieval_failed", False):
            return "END"
        return "summarizer"

    g.add_conditional_edges(
        "retriever",
        check_retrieval,
        {"summarizer": "summarizer", 
         "END": END
         }
    )
    g.add_edge("summarizer", "critic")
    g.add_edge("critic", "writer")
    g.add_edge("writer", END)

    g.set_entry_point("retriever")

    return g.compile()
