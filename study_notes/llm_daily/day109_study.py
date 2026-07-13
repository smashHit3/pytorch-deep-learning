"""Day 109: Compare a baseline and improved retrieval/reliability policy."""


def answer(query, documents, require_overlap):
    words = set(query.lower().split())
    source, text = max(documents.items(), key=lambda item: len(words & set(item[1].lower().split())))
    supported = bool(words & set(text.lower().split()))
    if require_overlap and not supported:
        return "I cannot answer from the retrieved notes."
    return f"[{source}] {text}"


def main():
    documents = {"attention": "attention uses queries keys and values"}
    cases = ["What does attention use?", "What is the launch date?"]
    for query in cases:
        baseline = answer(query, documents, require_overlap=False)
        improved = answer(query, documents, require_overlap=True)
        print({"query": query, "baseline": baseline, "improved": improved})
    print("Keep a fixed test set when judging whether a retrieval or reliability change actually helps.")

if __name__ == "__main__":
    main()
