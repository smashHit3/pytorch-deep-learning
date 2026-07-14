"""Day 43: Learn next-token prediction and autoregressive modeling.

A dependency-light, local demonstration for the Day 43 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

from collections import Counter

def main():
    # Adjacent corpus tokens become countable context-to-next-token training observations.
    corpus = "a tiny model predicts the next token a tiny model learns patterns".split()
    # zip stops at the shorter shifted sequence, creating exactly one ordered pair for each adjacent token pair.
    counts = Counter(zip(corpus[:-1], corpus[1:]))
    context = "tiny"
    # Filtering by "tiny" exposes only continuations observed after that exact one-token context.
    candidates = [(next_token, count) for (previous, next_token), count in counts.items() if previous == context]
    print("next-token candidates after", repr(context), ":", candidates)
    print("This count model is a transparent local baseline for an LM objective.")

# Shifting labels by one token converts a sequence into many next-token prediction training examples.
if __name__ == "__main__":
    main()
