"""Day 109: Compare a baseline and improved retrieval/reliability policy."""


def answer(query, documents, require_overlap):
    # Query words define the lexical evidence test used for both retrieval ranking and support verification.
    words = set(query.lower().split())
    source, text = max(documents.items(), key=lambda item: len(words & set(item[1].lower().split())))
    # The best document may still share no words, so the support check is separate from always selecting max().
    supported = bool(words & set(text.lower().split()))
    if require_overlap and not supported:
        return "I cannot answer from the retrieved notes."
    return f"[{source}] {text}"


def main():
    # One question overlaps the note and one does not, exposing the policy difference between baseline and abstention.
    documents = {"attention": "attention uses queries keys and values"}
    cases = ["What does attention use?", "What is the launch date?"]
    for query in cases:
        # Both policies retrieve identically; only the improved policy rejects a result with no lexical support.
        baseline = answer(query, documents, require_overlap=False)
        improved = answer(query, documents, require_overlap=True)
        print({"query": query, "baseline": baseline, "improved": improved})
    print("Keep a fixed test set when judging whether a retrieval or reliability change actually helps.")

# Comparing policies against a baseline shows whether added retrieval or reliability rules improve measured outcomes.
if __name__ == "__main__":
    main()
