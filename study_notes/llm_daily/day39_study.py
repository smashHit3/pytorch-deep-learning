"""Day 39: Study causal masking for autoregressive generation.

A dependency-light, local demonstration for the Day 39 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    scores = torch.ones(4, 4)
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    print(weights)
    assert torch.allclose(weights.masked_select(future), torch.zeros_like(weights.masked_select(future)))

if __name__ == "__main__":
    main()
