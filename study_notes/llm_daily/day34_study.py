"""Day 34: Study residual connections and layer normalization.

A dependency-light, local demonstration for the Day 34 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    x = torch.tensor([[1., 2., 3.]])
    update = torch.tensor([[.5, -.5, 1.]])
    residual = x + update
    normalized = torch.nn.LayerNorm(3)(residual)
    print("residual:", residual.tolist(), "mean after norm:", round(normalized.mean().item(), 6))

if __name__ == "__main__":
    main()
