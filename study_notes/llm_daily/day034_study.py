"""Day 34: Study residual connections and layer normalization.

A dependency-light, local demonstration for the Day 34 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    # The update has the same [1, 3] shape as x, which is required for an elementwise residual addition.
    x = torch.tensor([[1., 2., 3.]])
    update = torch.tensor([[.5, -.5, 1.]])
    # Residual addition retains each original feature while incorporating its corresponding transformed update.
    residual = x + update
    # LayerNorm normalizes the three features within each token rather than mixing different batch items.
    normalized = torch.nn.LayerNorm(3)(residual)
    print("residual:", residual.tolist(), "mean after norm:", round(normalized.mean().item(), 6))

# Residual paths preserve the original signal, and layer normalization stabilizes the scale seen by later layers.
if __name__ == "__main__":
    main()
