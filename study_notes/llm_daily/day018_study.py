"""Day 18: Train a small MLP on a simple dataset.

A dependency-light, local demonstration for the Day 18 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    torch.manual_seed(0)
    # These four binary inputs and labels form XOR, which requires a nonlinear hidden layer to separate.
    x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([[0.],[1.],[1.],[0.]])
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    # Adam repeatedly fits all four examples using logits, while BCE supplies gradients for the binary targets.
    for _ in range(200):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Thresholded probabilities are compared with the labels to report training-set accuracy, not generalization.
    predicted = (model(x).sigmoid() > .5).float()
    print("loss:", round(loss.item(), 4), "accuracy:", (predicted == y).float().mean().item())

# The loss-driven updates demonstrate that an MLP's hidden weights are learned jointly through backpropagation.
if __name__ == "__main__":
    main()
