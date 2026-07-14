"""Day 52: Add a causal attention layer and feed-forward block to a mini GPT."""

import torch


class TinyDecoderBlock(torch.nn.Module):
    """A compact pre-norm decoder block with causal self-attention."""

    def __init__(self, width):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(width, num_heads=2, batch_first=True)
        self.feed_forward = torch.nn.Sequential(torch.nn.Linear(width, 2 * width), torch.nn.ReLU(), torch.nn.Linear(2 * width, width))
        self.norm1, self.norm2 = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)

    def forward(self, hidden):
        length = hidden.size(1)
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal_mask, need_weights=False)
        hidden = self.norm1(hidden + attended)
        return self.norm2(hidden + self.feed_forward(hidden))


def main():
    hidden = torch.randn(1, 3, 6)
    output = TinyDecoderBlock(6)(hidden)
    print("input hidden shape:", tuple(hidden.shape))
    print("causal attention + feed-forward output shape:", tuple(output.shape))

if __name__ == "__main__":
    main()
