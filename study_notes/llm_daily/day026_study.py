"""Day 26: Learn LSTM and GRU at a high level.

A dependency-light, local demonstration for the Day 26 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    # The shared [1, 5, 3] input lets the LSTM and GRU expose comparable sequence-output shapes.
    x = torch.randn(1, 5, 3)
    # Both cells receive three features per step and emit four hidden features with batch-first layout.
    lstm = torch.nn.LSTM(3, 4, batch_first=True); gru = torch.nn.GRU(3, 4, batch_first=True)
    # Index zero selects the output at every time step; each architecture emits four hidden features.
    print("LSTM output shape:", tuple(lstm(x)[0].shape), "GRU output shape:", tuple(gru(x)[0].shape))
    print("Gates regulate what is written, retained, and exposed across time.")

# LSTM and GRU gates selectively retain or replace state, addressing some of a plain RNN's long-range limitations.
if __name__ == "__main__":
    main()
