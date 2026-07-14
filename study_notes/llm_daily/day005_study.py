"""Day 5: Study autograd; compute gradients for a tiny linear model.

A dependency-light, local demonstration for the Day 5 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    # These leaf scalars request gradients, representing the two trainable parameters of y = wx + b.
    weight = torch.tensor(2.0, requires_grad=True); bias = torch.tensor(1.0, requires_grad=True)
    # Squared error compares the single prediction at x=3 with the target value 10.
    prediction = weight * 3.0 + bias; loss = (prediction - 10.0) ** 2
    # backward applies the chain rule from this scalar loss to every reachable leaf requiring gradients.
    loss.backward()
    print("prediction/loss:", prediction.item(), loss.item())
    print("gradients (weight, bias):", weight.grad.item(), bias.grad.item())

# Calling backward accumulates derivatives on leaf tensors, exposing how the loss changes with the parameter.
if __name__ == "__main__":
    main()
