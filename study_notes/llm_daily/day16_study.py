"""Day 16: Study logistic regression and binary classification.

A dependency-light, local demonstration for the Day 16 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    features = torch.tensor([[-2.], [-1.], [1.], [2.]])
    labels = torch.tensor([[0.], [0.], [1.], [1.]])
    model = torch.nn.Linear(1, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.3)
    for _ in range(50):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    probabilities = model(features).sigmoid()
    print("probabilities:", probabilities.detach().squeeze().round(decimals=3).tolist())
    print("threshold .5 classes:", (probabilities > .5).int().squeeze().tolist())

if __name__ == "__main__":
    main()
