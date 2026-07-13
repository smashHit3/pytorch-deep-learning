"""Day 31: Implement single-head self-attention with small tensors.

A dependency-light, local demonstration for the Day 31 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math
import torch

def attention(query, key, value, mask=None):
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None: scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights

def main():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 4)
    output, weights = attention(x, x, x)
    print("input/output shapes:", tuple(x.shape), tuple(output.shape))
    print("attention rows sum to:", weights.sum(-1).round().tolist())
    print("first attention row:", weights[0, 0].tolist())

if __name__ == "__main__":
    main()
