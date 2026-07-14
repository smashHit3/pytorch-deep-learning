"""Day 25: Study RNN intuition and limitations.

A dependency-light, local demonstration for the Day 25 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    torch.manual_seed(0)
    sequence = torch.randn(1, 4, 3)
    rnn = torch.nn.RNN(3, 4, batch_first=True)
    output, hidden = rnn(sequence)
    print("RNN outputs:", tuple(output.shape), "final hidden:", tuple(hidden.shape))
    print("One recurrent state compresses prior context, which can make long dependencies difficult.")

if __name__ == "__main__":
    main()
