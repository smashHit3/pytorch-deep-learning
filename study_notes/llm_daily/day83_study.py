"""Day 83: Connect a retrieved context to a constrained local QA response."""


def retrieve(query, documents):
    query_words = set(query.lower().split())
    return max(documents.items(), key=lambda item: len(query_words & set(item[1].lower().split())))


def answer_from_context(source, context):
    """A transparent stand-in for an LLM that must cite supplied evidence."""
    return f"According to {source}: {context}" if context else "I do not have local evidence for that question."


def main():
    documents = {
        "attention-note": "Attention combines queries keys and values.",
        "retrieval-note": "Retrieval ranks local documents for a query.",
    }
    question = "How does retrieval rank documents?"
    source, context = retrieve(question, documents)
    print("question:", question)
    print("retrieved context:", {"source": source, "text": context})
    print("QA response:", answer_from_context(source, context))
    print("A real LLM generates from the retrieved prompt; attribution and abstention still need evaluation.")

if __name__ == "__main__":
    main()
