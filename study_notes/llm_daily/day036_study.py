"""Day 36: Study positional encoding and why order information is needed.

A dependency-light, local demonstration for the Day 36 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math

def positional_encoding(position, width=4):
    # Alternating sine and cosine frequencies give every integer position a distinct deterministic pattern.
    # The feature index selects the frequency, so low and high dimensions change at different rates with position.
    return [round(math.sin(position / (10000 ** (2*i/width))), 4) if i % 2 == 0 else round(math.cos(position / (10000 ** (2*(i//2)/width))), 4) for i in range(width)]

def main():
    # Comparing adjacent positions shows the order signal supplied alongside otherwise positionless embeddings.
    print("position 0:", positional_encoding(0))
    print("position 1:", positional_encoding(1))

# Positional encodings inject order because attention alone is permutation-equivariant over its token inputs.
if __name__ == "__main__":
    main()
