"""Day 14: Summarize backpropagation and optimization in your own words.

A dependency-light, local demonstration for the Day 14 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    # x is a leaf tensor, so backward stores the chain-rule derivative directly in x.grad.
    x = torch.tensor(3.0, requires_grad=True)
    # The nested expression makes this a compact example of derivatives through multiple operations.
    loss = (2 * x + 1) ** 2
    loss.backward()
    # At x=3, the displayed derivative evaluates the symbolic chain-rule expression shown in the output label.
    print("loss:", loss.item(), "d loss/dx:", x.grad.item(), "(chain rule: 2*(2x+1)*2)")

# Keeping this recap under the guard makes its printed study notes appear only when this lesson is run directly.
if __name__ == "__main__":
    main()
