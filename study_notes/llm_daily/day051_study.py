"""Day 51: Implement embeddings, positional embeddings, and input pipeline.

A dependency-light, local demonstration for the Day 51 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    token_embedding = torch.nn.Embedding(8, 6)
    position_embedding = torch.nn.Embedding(5, 6)
    ids = torch.tensor([[1, 2, 3]])
    positions = torch.arange(ids.size(1)).unsqueeze(0)
    hidden = token_embedding(ids) + position_embedding(positions)
    print("IDs:", tuple(ids.shape), "hidden:", tuple(hidden.shape), "logits:", tuple(torch.nn.Linear(6, 8)(hidden).shape))

if __name__ == "__main__":
    main()
