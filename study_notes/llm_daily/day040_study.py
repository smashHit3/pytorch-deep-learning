"""Day 40: Re-implement a Transformer block in PyTorch.

A dependency-light, local demonstration for the Day 40 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

class TinyBlock(torch.nn.Module):
    def __init__(self, width):
        # Two heads require width to divide evenly so each head receives an equal feature subspace.
        super().__init__(); self.attention = torch.nn.MultiheadAttention(width, 2, batch_first=True); self.norm1 = torch.nn.LayerNorm(width); self.ff = torch.nn.Sequential(torch.nn.Linear(width, 2*width), torch.nn.ReLU(), torch.nn.Linear(2*width, width)); self.norm2 = torch.nn.LayerNorm(width)
    def forward(self, x):
        # Self-attention uses the same [batch, tokens, width] tensor for queries, keys, and values.
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ff(x))

def main():
    # The test batch contains two three-token sequences, and the block preserves that external shape.
    x = torch.randn(2, 3, 4); print("block input/output:", tuple(x.shape), tuple(TinyBlock(4)(x).shape))

# A Transformer block alternates token mixing by attention with per-token nonlinear feature transformation.
if __name__ == "__main__":
    main()
