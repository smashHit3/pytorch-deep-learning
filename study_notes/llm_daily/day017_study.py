"""Day 17: Learn multilayer perceptrons and activation functions.

A dependency-light, local demonstration for the Day 17 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    # Fixing the random seed makes both randomly initialized comparison networks reproducible for this lesson.
    torch.manual_seed(0)
    # Three [1]-wide inputs expose how each activation transforms the same hidden linear features.
    x = torch.tensor([[-1.], [0.], [1.]])
    relu_model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    tanh_model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    # Detaching is only for display; it prevents the formatting path from retaining the computation graph.
    print("ReLU outputs:", relu_model(x).detach().squeeze().round(decimals=3).tolist())
    print("tanh outputs:", tanh_model(x).detach().squeeze().round(decimals=3).tolist())

# Nonlinear activations let stacked linear layers represent relationships that one linear transformation cannot.
if __name__ == "__main__":
    main()
