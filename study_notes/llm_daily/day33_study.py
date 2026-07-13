"""Day 33: Learn multi-head attention and why multiple heads help.

A dependency-light, local demonstration for the Day 33 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    x = torch.arange(8., dtype=torch.float32).reshape(1, 2, 4)
    heads = x.reshape(1, 2, 2, 2).transpose(1, 2)
    restored = heads.transpose(1, 2).reshape(1, 2, 4)
    print("input:", tuple(x.shape), "two heads:", tuple(heads.shape), "concatenated:", tuple(restored.shape))
    assert torch.equal(x, restored)

if __name__ == "__main__":
    main()
