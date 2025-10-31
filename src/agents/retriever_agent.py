import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_retriever_agent():
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def retrieve(topic: str, context: str, use_api: bool):
        if use_api:
            params = {
                "query": topic,
                "limit": 5,  # fetch top 10 papers
                "fields": "title,abstract,authors,year,url"
            }
            
            try:
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
            except Exception as e:
                return f"Error fetching data: {e}"

            if not data:
                retrieval_failed = True
                citations = "No Citations"
                return retrieval_failed, citations, "No papers found for this topic."

            summaries = []
            citations=""
            for paper in data:
                title = paper.get("title", "Unknown title")
                year = paper.get("year", "N/A")
                abstract = paper.get("abstract", "No abstract available")
                url = paper.get("url", "")
                summaries.append(f"📘 **{title} ({year})**\n{abstract}\n🔗 {url}\n")
                citations = citations + f"<ul><li><a href='{url}' target='_blank'>{title}</a> ({year}) – {', '.join([a['name'] for a in paper.get('authors', [])[:3]])}</li></ul><br>"

            retrieval_failed = False
        else:
            llm = ChatOpenAI(model="gpt-5-nano", temperature=0.3)
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = splitter.split_text(context)
            summaries = []

            for chunk in chunks:
                clean_chunk = " ".join(chunk.split())
                prompt = HumanMessage(content=f"""
                You are an academic research assistant.
                Given the following content from a research paper, generate a concise abstract (3-5 sentences):

                Content:
                {clean_chunk}

                Abstract:
                """)
            summary = llm.invoke([prompt])
            summaries.append(summary.content)
            citations="No Citations available"
            retrieval_failed = False
        
        return retrieval_failed, citations, "\n\n".join(summaries)

    return retrieve
