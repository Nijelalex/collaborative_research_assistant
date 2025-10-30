import requests

def get_retriever_agent():
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def retrieve(topic: str):
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

        return retrieval_failed, citations, "\n\n".join(summaries)

    return retrieve
