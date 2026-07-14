"""Day 51: Implement embeddings, positional embeddings, and input pipeline.

A dependency-light, local demonstration for the Day 51 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    # Both tables use width six so token identity and absolute position vectors can be added elementwise.
    token_embedding = torch.nn.Embedding(8, 6)
    position_embedding = torch.nn.Embedding(5, 6)
    ids = torch.tensor([[1, 2, 3]])
    # A [1, 3] position row aligns with the [1, 3] ID batch and is looked up once per token location.
    positions = torch.arange(ids.size(1)).unsqueeze(0)
    hidden = token_embedding(ids) + position_embedding(positions)
    # The final linear layer preserves batch and sequence axes while replacing width with vocabulary size eight.
    print("IDs:", tuple(ids.shape), "hidden:", tuple(hidden.shape), "logits:", tuple(torch.nn.Linear(6, 8)(hidden).shape))

# Token and positional embeddings are added because each position needs both identity and order information.
if __name__ == "__main__":
    main()
