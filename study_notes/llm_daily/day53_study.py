"""Day 53: Add training loop, optimizer, and evaluation.

A dependency-light, local demonstration for the Day 53 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    train = torch.tensor([0, 1, 0, 1]); valid = torch.tensor([0, 1])
    model = torch.nn.Embedding(2, 2); head = torch.nn.Linear(2, 2); optimizer = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=.2)
    best = float('inf')
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(head(model(train[:-1])), train[1:]); optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad(): best = min(best, torch.nn.functional.cross_entropy(head(model(valid[:-1])), valid[1:]).item())
    print("training loss:", round(loss.item(), 4), "best validation loss:", round(best, 4))

if __name__ == "__main__":
    main()
