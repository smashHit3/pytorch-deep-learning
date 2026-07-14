"""Day 92: Learn the goal of RLHF and preference optimization.

A dependency-light, local demonstration for the Day 92 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # Each candidate has separate toy helpfulness and safety values so their policy tradeoff is visible.
    candidates = {"helpful_grounded": (2, 2), "fluent_unsupported": (2, 0), "refusal": (0, 2)}
    # A toy preference score; actual RLHF uses human preference data and optimization machinery.
    # Summing dimensions creates the displayed ordering but cannot encode real preference judgments or edge cases.
    ranked = sorted(((helpful + safe, name) for name, (helpful, safe) in candidates.items()), reverse=True)
    print("preference ranking:", ranked)
    # This output explains the policy goal, while the scores remain deliberately transparent teaching data.
    print("Preference optimization aims to make behavior more helpful while respecting safety tradeoffs.")

# Summing helpfulness and safety creates a transparent toy ranking, not a substitute for learned human preferences.
if __name__ == "__main__":
    main()
