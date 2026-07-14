"""Day 8: Review vectors, matrices, dot products, and matrix multiplication.

A dependency-light, local demonstration for the Day 8 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    # The matrix has one row per output coordinate and three columns to match the vector width.
    vector = torch.tensor([1.0, 2.0, 3.0])
    # The first row weights vector coordinates one and three, while the second weights coordinates two and three.
    matrix = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    # dot reduces two vectors to one similarity-like scalar, whereas @ retains one result per matrix row.
    print("dot:", torch.dot(vector, vector).item())
    print("matrix @ vector:", (matrix @ vector).tolist())

# A dot product collapses matching coordinates to one score, whereas matrix multiplication combines rows and columns.
if __name__ == "__main__":
    main()
