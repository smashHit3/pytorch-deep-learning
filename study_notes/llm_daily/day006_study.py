"""Day 6: Build a minimal training loop with dummy data.

A dependency-light, local demonstration for the Day 6 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    torch.manual_seed(0)
    # Inputs have shape [5, 1], and the exact target rule gives the one-feature model a learnable signal.
    x = torch.arange(1., 6.).unsqueeze(1)
    target = 2 * x + 1
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.01)
    # SGD updates the linear layer from the full five-example batch on every iteration.
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(model(x), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # The final loss reports fit on training data; the extrapolated x=6 prediction is not a validation metric.
    print("x dtype/device:", x.dtype, x.device, "final loss:", round(loss.item(), 5))
    print("prediction at 6:", round(model(torch.tensor([[6.]])).item(), 3))

# Each training iteration clears old gradients before backpropagation so updates use only the current loss.
if __name__ == "__main__":
    main()
