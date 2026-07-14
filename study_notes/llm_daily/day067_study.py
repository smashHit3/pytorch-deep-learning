"""Day 67: Study QLoRA and quantized fine-tuning concepts.

A dependency-light, local demonstration for the Day 67 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import torch

def main():
    torch.manual_seed(0)
    base = torch.randn(4, 4)
    left, right = torch.randn(4, 1), torch.randn(1, 4)
    adapted = base + left @ right
    scale = 127 / base.abs().max(); quantized = (base * scale).round().clamp(-127, 127); restored = quantized / scale
    print("base parameters:", base.numel(), "adapter parameters:", left.numel()+right.numel())
    print("low-rank update norm:", round((adapted-base).norm().item(), 4), "quantization MAE:", round((base-restored).abs().mean().item(), 5))

if __name__ == "__main__":
    main()
