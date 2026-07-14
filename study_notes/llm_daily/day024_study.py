"""Day 24: Learn embeddings and why token IDs need vector representations.

A dependency-light, local demonstration for the Day 24 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    # The table maps five valid IDs to three-dimensional trainable vectors.
    embedding = torch.nn.Embedding(5, 3)
    # Looking up three IDs returns one vector per ID, hence the displayed [3, 3] shape.
    ids = torch.tensor([0, 3, 1])
    print("IDs:", ids.tolist(), "embedding shape:", tuple(embedding(ids).shape))
    # Index zero chooses the first requested ID's vector, not necessarily vocabulary ID zero.
    print("first vector:", embedding(ids)[0].detach().tolist())

# Embeddings replace arbitrary token IDs with trainable vectors whose geometry can encode useful similarity.
if __name__ == "__main__":
    main()
