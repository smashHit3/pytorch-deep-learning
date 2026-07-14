"""Day 32: Calculate scaled dot-product attention one operation at a time."""

import math
import torch


def main():
    # One query attends over two width-two key and value vectors, producing a single width-two output.
    query = torch.tensor([[1.0, 2.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
    raw_scores = query @ keys.T
    # Dividing by sqrt(d_k) limits logit magnitude before softmax compares the two keys.
    scaled_scores = raw_scores / math.sqrt(query.size(-1))
    weights = torch.softmax(scaled_scores, dim=-1)
    # Weighted aggregation retains the value width while blending rows according to the normalized scores.
    output = weights @ values
    print("QK^T scores:", raw_scores.tolist())
    print("scaled by sqrt(d_k):", scaled_scores.tolist())
    print("softmax weights:", weights.tolist(), "sum:", weights.sum().item())
    print("weighted values:", output.tolist())

# The mask replaces disallowed scores before softmax, making their resulting attention probability effectively zero.
if __name__ == "__main__":
    main()
