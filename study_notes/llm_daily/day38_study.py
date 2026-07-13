"""Day 38: Learn feed-forward blocks and attention masks.

A dependency-light, local demonstration for the Day 38 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    hidden = torch.tensor([[1., 2., 3.]])
    ff = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    scores = torch.tensor([[0., 0., 0.]])
    padding_mask = torch.tensor([[False, False, True]])
    print("feed-forward output shape:", tuple(ff(hidden).shape))
    print("masked attention weights:", torch.softmax(scores.masked_fill(padding_mask, float('-inf')), dim=-1).tolist())

if __name__ == "__main__":
    main()
