from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def get_summarizer_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def summarize(text):
        prompt = HumanMessage(content=f"Summarize the following academic content concisely:\n\n{text}\n\nSummary:")
        return llm.invoke([prompt])
    
    return summarize
