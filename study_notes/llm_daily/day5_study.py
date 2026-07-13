"""Day 5: Study autograd; compute gradients for a tiny linear model.

A dependency-light, local demonstration for the Day 5 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    weight = torch.tensor(2.0, requires_grad=True); bias = torch.tensor(1.0, requires_grad=True)
    prediction = weight * 3.0 + bias; loss = (prediction - 10.0) ** 2
    loss.backward()
    print("prediction/loss:", prediction.item(), loss.item())
    print("gradients (weight, bias):", weight.grad.item(), bias.grad.item())

if __name__ == "__main__":
    main()
