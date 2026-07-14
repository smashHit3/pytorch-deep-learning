"""Day 46: Train a tiny language model on a small corpus.

A dependency-light, local demonstration for the Day 46 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    torch.manual_seed(0)
    tokens = torch.tensor([0, 1, 2, 0, 1, 2])
    inputs, targets = tokens[:-1], tokens[1:]
    model = torch.nn.Sequential(torch.nn.Embedding(3, 4), torch.nn.Linear(4, 3)); optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    for _ in range(80):
        loss = torch.nn.functional.cross_entropy(model(inputs), targets); optimizer.zero_grad(); loss.backward(); optimizer.step()
    print("tiny LM loss:", round(loss.item(), 4), "predicted next IDs:", model(inputs).argmax(-1).tolist())

if __name__ == "__main__":
    main()
