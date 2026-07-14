"""Day 39: Study causal masking for autoregressive generation.

A dependency-light, local demonstration for the Day 39 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    # The strict upper triangle marks keys later than each query position in this four-token sequence.
    scores = torch.ones(4, 4)
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    # Rows are query positions; their progressively longer visible prefixes are visible in the printed matrix.
    print(weights)
    # The assertion checks the causal invariant: every masked future probability is exactly zero.
    assert torch.allclose(weights.masked_select(future), torch.zeros_like(weights.masked_select(future)))

# A causal mask blocks future keys, ensuring a next-token model cannot inspect the answer it is meant to predict.
if __name__ == "__main__":
    main()
