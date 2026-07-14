"""Day 55: Debug tensor shapes, a causal mask, and sampling assumptions."""

import torch


def main():
    batch, tokens, width, heads = 2, 4, 8, 2
    scores = torch.ones(batch, heads, tokens, tokens)
    future = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
    masked_scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    print("Q/K score shape:", tuple(scores.shape), "expected:", (batch, heads, tokens, tokens))
    print("future probability is zero:", bool(torch.all(weights.masked_select(future) == 0)))
    print("sampling check: temperature must be positive; greedy decoding uses argmax.")

if __name__ == "__main__":
    main()
