"""Day 24: Learn embeddings and why token IDs need vector representations.

A dependency-light, local demonstration for the Day 24 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    embedding = torch.nn.Embedding(5, 3)
    ids = torch.tensor([0, 3, 1])
    print("IDs:", ids.tolist(), "embedding shape:", tuple(embedding(ids).shape))
    print("first vector:", embedding(ids)[0].detach().tolist())

if __name__ == "__main__":
    main()
