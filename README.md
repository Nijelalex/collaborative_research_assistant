# Research Assistant

Build a multi-agent AI “research team” that:

# Literature Review Pipeline

**User Topic**  

**Retriever Agent**  
- Uses Semantic Scholar API  
- Retrieves abstracts  
- Builds RAG context  

**Summarizer Agent**  
- Summarizes top 10 abstracts         ↓  

**Critic Agent**  
- Evaluates quality of summaries  
- Identifies gaps  

**Writer Agent**  
- Synthesizes literature review  

### To do
 
RAG, save by topic and tags to vector db
topic searched previously to be seen

Create session for each user, save memory (UI also changes)

Create error 