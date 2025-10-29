from langchain_openai import ChatOpenAI

def get_writer_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    def write_review(summary, critique):
        prompt = (
            "Using the following summary and critique, synthesize a coherent literature "
            "review paragraph suitable for a research paper.\n\n"
            f"Summary:\n{summary}\n\nCritique:\n{critique}\n\nLiterature Review:"
        )
        return llm.predict(prompt)
    
    return write_review
