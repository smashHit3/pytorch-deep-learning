"""Day 16: Study logistic regression and binary classification.

A dependency-light, local demonstration for the Day 16 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    # Negative and positive one-dimensional features map to binary labels with shape [4, 1].
    features = torch.tensor([[-2.], [-1.], [1.], [2.]])
    labels = torch.tensor([[0.], [0.], [1.], [1.]])
    model = torch.nn.Linear(1, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.3)
    # BCE-with-logits consumes unbounded scores; keeping sigmoid outside the loss avoids numerical instability.
    for _ in range(50):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # A 0.5 threshold turns calibrated sigmoid outputs into the printed hard class decisions.
    probabilities = model(features).sigmoid()
    print("probabilities:", probabilities.detach().squeeze().round(decimals=3).tolist())
    print("threshold .5 classes:", (probabilities > .5).int().squeeze().tolist())

# Logistic regression maps a linear score through a sigmoid so the output can be interpreted as a binary probability.
if __name__ == "__main__":
    main()
