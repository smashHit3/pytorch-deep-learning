"""Day 84: Evaluate retrieval quality and improve chunking strategy.

A dependency-light, local demonstration for the Day 84 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

# Recall@k is true when at least one labeled-relevant source appears among the first k results.
def recall_at_k(ranked, relevant, k): return bool(set(ranked[:k]) & set(relevant))

def main():
    # Each case supplies a ranking and independent relevance label, which retrieval evaluation requires.
    cases = [(["a", "b", "c"], {"b"}), (["x", "y", "z"], {"z"})]
    # Increasing k trades a longer context candidate set for greater chance of including relevant evidence.
    for k in (1, 2, 3): print(f"recall@{k}:", sum(recall_at_k(r, rel, k) for r,rel in cases) / len(cases))
    print("Change chunk boundaries, then remeasure with labeled relevance rather than guessing.")

# Retrieval evaluation separates whether the right source was found from whether a later model used it well.
if __name__ == "__main__":
    main()
