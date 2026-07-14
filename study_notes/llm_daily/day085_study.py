"""Day 85: Trace the components and information flow of a RAG architecture."""


def main():
    # The stages deliberately separate offline document indexing from per-question retrieval and generation.
    stages = [
        ("offline indexing", "documents -> chunks -> embeddings -> vector store"),
        ("query handling", "question -> query embedding -> top-k retrieval"),
        ("context construction", "retrieved chunks + citations -> prompt within token budget"),
        ("generation", "LLM prompt -> answer that can abstain when evidence is missing"),
        ("evaluation", "answer, citations, and retrieval are checked separately"),
    ]
    # Context construction includes citations and a token budget because retrieved text cannot all fit blindly.
    # Iterating tuple pairs keeps each named RAG stage attached to its corresponding information-flow description.
    for stage, flow in stages:
        print(f"{stage}: {flow}")
    print("RAG augments a model with selected external context; it does not make every answer correct.")

# The RAG flow keeps retrieval and generation distinct: documents supply evidence before the model composes an answer.
if __name__ == "__main__":
    main()
