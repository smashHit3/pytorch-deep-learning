"""Day 31: Implement single-head self-attention with small tensors.

A dependency-light, local demonstration for the Day 31 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math
import torch

def attention(query, key, value, mask=None):
    # The matrix product creates [batch, query_tokens, key_tokens] attention logits.
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    # Applying any causal or padding mask before softmax prevents forbidden values from affecting the read.
    if mask is not None: scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights

def main():
    torch.manual_seed(0)
    # Identical Q, K, and V inputs implement a single self-attention head for a three-token sequence.
    x = torch.randn(1, 3, 4)
    output, weights = attention(x, x, x)
    print("input/output shapes:", tuple(x.shape), tuple(output.shape))
    print("attention rows sum to:", weights.sum(-1).round().tolist())
    print("first attention row:", weights[0, 0].tolist())

# Scaling dot-product scores by sqrt(d) keeps softmax from becoming overly sharp as key dimension grows.
if __name__ == "__main__":
    main()
