"""Day 55: Debug tensor shapes, a causal mask, and sampling assumptions."""

import torch


def main():
    # Scores have a query and key token axis for every batch item and attention head.
    batch, tokens, width, heads = 2, 4, 8, 2
    scores = torch.ones(batch, heads, tokens, tokens)
    # The strict upper triangle marks keys that would reveal future next-token labels to a query.
    future = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
    masked_scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    print("Q/K score shape:", tuple(scores.shape), "expected:", (batch, heads, tokens, tokens))
    print("future probability is zero:", bool(torch.all(weights.masked_select(future) == 0)))
    # A positive temperature is required for meaningful logit division before softmax sampling.
    print("sampling check: temperature must be positive; greedy decoding uses argmax.")

# Debugging the mask and tensor shapes prevents accidental future-token leakage that can make toy results misleading.
if __name__ == "__main__":
    main()
