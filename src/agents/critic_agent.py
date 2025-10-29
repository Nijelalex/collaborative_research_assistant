from langchain_openai import ChatOpenAI

def get_critic_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def critique(summary):
        prompt = (
            "Critically evaluate the following research summary: "
            "identify strengths, weaknesses, and potential research gaps.\n\n"
            f"{summary}\n\nCritique:"
        )
        return llm.predict(prompt)
    
    return critique
