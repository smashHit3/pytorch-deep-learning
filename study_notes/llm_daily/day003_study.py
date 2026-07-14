"""Day 3: Practice NumPy-style array shape operations with local torch tensors.

Torch is used because the study materials intentionally require no NumPy dependency;
the reshape, transpose, slice, and broadcasting semantics are the same ideas.
"""

import torch


def main():
    # arange creates six ordered values, and reshape groups them into two rows of three without copying.
    array = torch.arange(1, 7).reshape(2, 3)
    # Each single-column offset is repeated across its matching array row during broadcasting.
    row_offsets = torch.tensor([[10], [20]])
    print("shape:", tuple(array.shape), "array:", array.tolist())
    print("reshape (3, 2):", array.reshape(3, 2).tolist())
    print("transpose:", array.transpose(0, 1).tolist())
    print("slice first column:", array[:, 0].tolist())
    # A [2, 1] offset broadcasts across the three columns of each row in the [2, 3] array.
    print("broadcast row offsets:", (array + row_offsets).tolist())

# Reshape changes how values are grouped without changing their order or the total number of elements.
if __name__ == "__main__":
    main()
