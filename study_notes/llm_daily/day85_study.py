"""Day 85: Trace the components and information flow of a RAG architecture."""


def main():
    stages = [
        ("offline indexing", "documents -> chunks -> embeddings -> vector store"),
        ("query handling", "question -> query embedding -> top-k retrieval"),
        ("context construction", "retrieved chunks + citations -> prompt within token budget"),
        ("generation", "LLM prompt -> answer that can abstain when evidence is missing"),
        ("evaluation", "answer, citations, and retrieval are checked separately"),
    ]
    for stage, flow in stages:
        print(f"{stage}: {flow}")
    print("RAG augments a model with selected external context; it does not make every answer correct.")

if __name__ == "__main__":
    main()
