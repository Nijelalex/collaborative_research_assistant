from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def get_critic_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    def critique(summary):
        prompt = HumanMessage(content=
            "Critically evaluate the following research summary: "
            "identify strengths, weaknesses, and potential research gaps providing 2 points for each.\n\n"
            f"{summary}\n\nCritique:"
        )
        return llm.invoke([prompt])
    
    return critique
