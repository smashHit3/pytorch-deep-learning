"""Day 36: Study positional encoding and why order information is needed.

A dependency-light, local demonstration for the Day 36 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math

def positional_encoding(position, width=4):
    return [round(math.sin(position / (10000 ** (2*i/width))), 4) if i % 2 == 0 else round(math.cos(position / (10000 ** (2*(i//2)/width))), 4) for i in range(width)]

def main():
    print("position 0:", positional_encoding(0))
    print("position 1:", positional_encoding(1))

if __name__ == "__main__":
    main()
