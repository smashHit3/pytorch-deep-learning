"""Day 14: Summarize backpropagation and optimization in your own words.

A dependency-light, local demonstration for the Day 14 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    x = torch.tensor(3.0, requires_grad=True)
    loss = (2 * x + 1) ** 2
    loss.backward()
    print("loss:", loss.item(), "d loss/dx:", x.grad.item(), "(chain rule: 2*(2x+1)*2)")

if __name__ == "__main__":
    main()
