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
            return "No papers found for this topic."

        summaries = []
        for paper in data:
            title = paper.get("title", "Unknown title")
            year = paper.get("year", "N/A")
            abstract = paper.get("abstract", "No abstract available")
            url = paper.get("url", "")
            summaries.append(f"📘 **{title} ({year})**\n{abstract}\n🔗 {url}\n")

        return "\n\n".join(summaries)

    return retrieve
