"""Day 17: Learn multilayer perceptrons and activation functions.

A dependency-light, local demonstration for the Day 17 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    torch.manual_seed(0)
    x = torch.tensor([[-1.], [0.], [1.]])
    relu_model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    tanh_model = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1))
    print("ReLU outputs:", relu_model(x).detach().squeeze().round(decimals=3).tolist())
    print("tanh outputs:", tanh_model(x).detach().squeeze().round(decimals=3).tolist())

if __name__ == "__main__":
    main()
