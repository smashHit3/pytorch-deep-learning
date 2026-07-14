"""Day 25: Study RNN intuition and limitations.

A dependency-light, local demonstration for the Day 25 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    torch.manual_seed(0)
    # batch_first=True interprets this input as [batch=1, time=4, features=3].
    sequence = torch.randn(1, 4, 3)
    # The RNN maps three input features at each time step into a hidden state of width four.
    rnn = torch.nn.RNN(3, 4, batch_first=True)
    output, hidden = rnn(sequence)
    # output retains a hidden vector for every time step; hidden contains only the final recurrent state.
    print("RNN outputs:", tuple(output.shape), "final hidden:", tuple(hidden.shape))
    print("One recurrent state compresses prior context, which can make long dependencies difficult.")

# An RNN carries a hidden state forward, but repeated sequential updates make distant information hard to preserve.
if __name__ == "__main__":
    main()
