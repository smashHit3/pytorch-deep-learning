"""Day 4: Inspect PyTorch tensor dtype, device, and basic math operations."""

import torch


def main():
    integers = torch.tensor([1, 2, 3], dtype=torch.int64)
    values = integers.to(dtype=torch.float32)
    other = torch.tensor([0.5, 1.5, 2.5])
    print("integer tensor:", integers.tolist(), "dtype:", integers.dtype, "device:", integers.device)
    print("float tensor:", values.tolist(), "dtype:", values.dtype)
    print("add:", (values + other).tolist(), "multiply:", (values * other).tolist())
    print("mean:", values.mean().item(), "matrix product:", (values.unsqueeze(0) @ other.unsqueeze(1)).item())

if __name__ == "__main__":
    main()
