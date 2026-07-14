"""Day 8: Review vectors, matrices, dot products, and matrix multiplication.

A dependency-light, local demonstration for the Day 8 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    print("dot:", torch.dot(vector, vector).item())
    print("matrix @ vector:", (matrix @ vector).tolist())

if __name__ == "__main__":
    main()
