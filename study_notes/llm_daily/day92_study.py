"""Day 92: Learn the goal of RLHF and preference optimization.

A dependency-light, local demonstration for the Day 92 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    candidates = {"helpful_grounded": (2, 2), "fluent_unsupported": (2, 0), "refusal": (0, 2)}
    # A toy preference score; actual RLHF uses human preference data and optimization machinery.
    ranked = sorted(((helpful + safe, name) for name, (helpful, safe) in candidates.items()), reverse=True)
    print("preference ranking:", ranked)
    print("Preference optimization aims to make behavior more helpful while respecting safety tradeoffs.")

if __name__ == "__main__":
    main()
