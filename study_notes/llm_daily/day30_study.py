"""Day 30: Learn query, key, and value intuition with a tiny example.

A dependency-light, local demonstration for the Day 30 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    query = torch.tensor([1., 0.]); keys = torch.tensor([[1., 0.], [0., 1.]])
    values = torch.tensor([[10., 0.], [0., 20.]])
    weights = torch.softmax(keys @ query, dim=0); read = weights @ values
    print("query-key scores:", (keys @ query).tolist(), "weights:", weights.tolist())
    print("weighted value read:", read.tolist())

if __name__ == "__main__":
    main()
