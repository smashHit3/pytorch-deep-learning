"""Day 15: Study linear regression and implement it in PyTorch.

A dependency-light, local demonstration for the Day 15 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    torch.manual_seed(0)
    # The [5, 1] feature matrix and y=2x+1 target identify the slope and intercept being learned.
    x = torch.arange(1., 6.).unsqueeze(1)
    target = 2 * x + 1
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.01)
    # Each full-batch pass computes MSE, clears accumulated gradients, and applies one regularized SGD update.
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(model(x), target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    # The last print distinguishes observed training error from the model's extrapolation at an unseen input.
    print("x dtype/device:", x.dtype, x.device, "final loss:", round(loss.item(), 5))
    print("prediction at 6:", round(model(torch.tensor([[6.]])).item(), 3))

# Linear regression learns a slope and intercept by minimizing prediction error over the observed examples.
if __name__ == "__main__":
    main()
