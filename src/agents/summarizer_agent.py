from langchain_openai import ChatOpenAI

def get_summarizer_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def summarize(text):
        prompt = f"Summarize the following academic content concisely:\n\n{text}\n\nSummary:"
        return llm.predict(prompt)
    
    return summarize
