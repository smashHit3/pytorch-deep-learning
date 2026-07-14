"""Day 29: Study the problem attention is trying to solve.

A dependency-light, local demonstration for the Day 29 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math
import torch

def attention(query, key, value, mask=None):
    # QK^T yields one compatibility score for every query-key token pair in the sequence.
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    # Masked positions receive negative infinity so softmax assigns them zero probability.
    if mask is not None: scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights

def main():
    torch.manual_seed(0)
    # Reusing x for Q, K, and V makes this self-attention over three tokens with width four.
    x = torch.randn(1, 3, 4)
    output, weights = attention(x, x, x)
    print("input/output shapes:", tuple(x.shape), tuple(output.shape))
    print("attention rows sum to:", weights.sum(-1).round().tolist())
    print("first attention row:", weights[0, 0].tolist())

# Attention forms a content-dependent weighted summary, so a token can emphasize the most relevant context.
if __name__ == "__main__":
    main()
