from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from db import search_similar_rag

def generate_answer(query, top_k=3):
    top_results = search_similar_rag(query, top_k)
    if not top_results:
        return "⚠️ No data available to answer the question."

    combined_text = ""
    for _, r in top_results:
        combined_text += f"Topic: {r['topic']}\nSummary: {r['summary']}\nFinal: {r['final']}\n\n"

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    prompt = HumanMessage(content=f"""
                You are a research assistant.
                Answer the following question based on the retrieved research content. Provide a 200 words answer.
                If the query is not able to get good answer from research content then tell that to student.
                          
                Question: {query}

                Retrieved Research Content:
                {combined_text}

                Answer:
                """)
    answer = llm.invoke([prompt])
    return answer.content
