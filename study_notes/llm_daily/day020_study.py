"""Day 20: Learn regularization, dropout, and weight decay.

A dependency-light, local demonstration for the Day 20 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    torch.manual_seed(0)
    # The same [3, 4] input is run in both modes to isolate dropout's train-versus-evaluation behavior.
    x = torch.ones(3, 4); layer = torch.nn.Linear(4, 4); dropout = torch.nn.Dropout(.5)
    # train() enables random activation removal; eval() makes the dropout module pass activations through.
    layer.train(); train_output = dropout(layer(x)); layer.eval(); eval_output = dropout(layer(x))
    # Squaring and summing every weight and bias produces the unscaled L2 term used by weight decay.
    penalty = sum(parameter.square().sum() for parameter in layer.parameters())
    print("dropout zeros in train mode:", int((train_output == 0).sum()), "in eval mode:", int((eval_output == 0).sum()))
    print("L2 penalty:", round(penalty.item(), 4), "(weight decay adds a scaled version to the loss)")

# Dropout perturbs training-time activations, while weight decay discourages unnecessarily large parameters.
if __name__ == "__main__":
    main()
