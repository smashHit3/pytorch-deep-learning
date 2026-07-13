"""Day 26: Learn LSTM and GRU at a high level.

A dependency-light, local demonstration for the Day 26 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    x = torch.randn(1, 5, 3)
    lstm = torch.nn.LSTM(3, 4, batch_first=True); gru = torch.nn.GRU(3, 4, batch_first=True)
    print("LSTM output shape:", tuple(lstm(x)[0].shape), "GRU output shape:", tuple(gru(x)[0].shape))
    print("Gates regulate what is written, retained, and exposed across time.")

if __name__ == "__main__":
    main()
