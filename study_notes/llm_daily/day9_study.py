"""Day 9: Use derivatives and partial derivatives to take gradient-descent steps."""

import torch


def main():
    parameters = torch.tensor([0.0, 0.0], requires_grad=True)  # slope and intercept
    x, targets = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 5.0])
    prediction = parameters[0] * x + parameters[1]
    loss = ((prediction - targets) ** 2).mean()
    loss.backward()
    learning_rate = 0.1
    updated = parameters.detach() - learning_rate * parameters.grad
    print("loss:", round(loss.item(), 3))
    print("partial derivatives [slope, intercept]:", parameters.grad.tolist())
    print("one gradient-descent update:", updated.tolist())

if __name__ == "__main__":
    main()
