"""Day 66: Show LoRA as a trainable low-rank update to frozen base weights."""

import torch


def main():
    # A fixed seed makes the random base and adapter matrices reproducible when comparing parameter counts.
    torch.manual_seed(0)
    # The 4-by-4 base matrix represents frozen pretrained weights, while rank one limits adapter capacity.
    base_weight = torch.randn(4, 4)
    rank = 1
    lora_a, lora_b = torch.randn(4, rank), torch.randn(rank, 4)
    # Multiplying [4, 1] by [1, 4] produces a full-shape but low-rank delta for the base weight.
    adapted_weight = base_weight + lora_a @ lora_b
    print("base parameters:", base_weight.numel())
    print("LoRA parameters at rank 1:", lora_a.numel() + lora_b.numel())
    print("base frozen:", not base_weight.requires_grad, "adapter update norm:", round((adapted_weight - base_weight).norm().item(), 4))

# LoRA trains a low-rank delta while freezing base weights, reducing the number of parameters that need gradients.
if __name__ == "__main__":
    main()
