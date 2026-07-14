"""Day 4: Inspect PyTorch tensor dtype, device, and basic math operations."""

import torch


def main():
    # Converting int64 values to float32 permits floating-point arithmetic and mean reduction.
    integers = torch.tensor([1, 2, 3], dtype=torch.int64)
    values = integers.to(dtype=torch.float32)
    # This float tensor has the same length as values, enabling elementwise addition and multiplication.
    other = torch.tensor([0.5, 1.5, 2.5])
    print("integer tensor:", integers.tolist(), "dtype:", integers.dtype, "device:", integers.device)
    print("float tensor:", values.tolist(), "dtype:", values.dtype)
    print("add:", (values + other).tolist(), "multiply:", (values * other).tolist())
    # unsqueeze turns two length-3 vectors into [1, 3] and [3, 1] matrices for a scalar dot product.
    print("mean:", values.mean().item(), "matrix product:", (values.unsqueeze(0) @ other.unsqueeze(1)).item())

# Dtype and device travel with a tensor, so arithmetic results reflect both representation and placement.
if __name__ == "__main__":
    main()
