"""Day 3: Practice NumPy-style array shape operations with local torch tensors.

Torch is used because the study materials intentionally require no NumPy dependency;
the reshape, transpose, slice, and broadcasting semantics are the same ideas.
"""

import torch


def main():
    array = torch.arange(1, 7).reshape(2, 3)
    row_offsets = torch.tensor([[10], [20]])
    print("shape:", tuple(array.shape), "array:", array.tolist())
    print("reshape (3, 2):", array.reshape(3, 2).tolist())
    print("transpose:", array.transpose(0, 1).tolist())
    print("slice first column:", array[:, 0].tolist())
    print("broadcast row offsets:", (array + row_offsets).tolist())

if __name__ == "__main__":
    main()
