"""Day 87: Improve the retriever or prompt template in your RAG app.

A dependency-light, local demonstration for the Day 87 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def rank(query, docs):
    query_words = set(query.lower().split())
    return sorted(((len(query_words & set(text.lower().split())), name) for name,text in docs.items()), reverse=True)

def main():
    docs = {"attention": "keys values attention", "retrieval": "retrieval ranks documents"}
    query = "how retrieval ranks documents"
    print("baseline ranking:", rank(query, docs))
    print("improvement idea: normalize words or rewrite the query, then compare on a fixed test set.")

if __name__ == "__main__":
    main()
