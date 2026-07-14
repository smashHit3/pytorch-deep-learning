"""Day 44: Prepare a tiny text dataset for language modeling.

A dependency-light, local demonstration for the Day 44 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

from collections import Counter

def main():
    # Whitespace tokens are assigned sorted IDs so the printed vocabulary is deterministic across runs.
    text = "tiny models learn from tiny local corpora"
    tokens = text.split()
    vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}
    # Repeated tokens reuse their vocabulary ID, preserving the original sequence's repetitions in encoded form.
    encoded = [vocab[t] for t in tokens]
    print("vocabulary:", vocab)
    # The shifted pairs use each ID as context for its immediate successor, the basic next-token objective.
    print("encoded:", encoded, "shifted pairs:", list(zip(encoded[:-1], encoded[1:])))

# Language-model examples pair each context window with the same window shifted one position to the right.
if __name__ == "__main__":
    main()
