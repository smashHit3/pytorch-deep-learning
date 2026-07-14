"""Day 32: Calculate scaled dot-product attention one operation at a time."""

import math
import torch


def main():
    query = torch.tensor([[1.0, 2.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
    raw_scores = query @ keys.T
    scaled_scores = raw_scores / math.sqrt(query.size(-1))
    weights = torch.softmax(scaled_scores, dim=-1)
    output = weights @ values
    print("QK^T scores:", raw_scores.tolist())
    print("scaled by sqrt(d_k):", scaled_scores.tolist())
    print("softmax weights:", weights.tolist(), "sum:", weights.sum().item())
    print("weighted values:", output.tolist())

if __name__ == "__main__":
    main()
