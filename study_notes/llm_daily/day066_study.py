"""Day 66: Show LoRA as a trainable low-rank update to frozen base weights."""

import torch


def main():
    torch.manual_seed(0)
    base_weight = torch.randn(4, 4)
    rank = 1
    lora_a, lora_b = torch.randn(4, rank), torch.randn(rank, 4)
    adapted_weight = base_weight + lora_a @ lora_b
    print("base parameters:", base_weight.numel())
    print("LoRA parameters at rank 1:", lora_a.numel() + lora_b.numel())
    print("base frozen:", not base_weight.requires_grad, "adapter update norm:", round((adapted_weight - base_weight).norm().item(), 4))

if __name__ == "__main__":
    main()
