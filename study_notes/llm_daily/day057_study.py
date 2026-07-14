"""Day 57: Study causal LM versus masked LM objectives.

A dependency-light, local demonstration for the Day 57 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

import math

def main():
    # Causal targets pair each prefix position with the token immediately to its right.
    tokens = ["the", "small", "model", "learns"]
    causal_targets = list(enumerate(tokens[1:], start=0))
    # A masked-LM label remains at its original position while surrounding left and right tokens stay visible.
    masked_position = 2
    # Enumerated causal positions index the token supplying context, while each paired label is its successor.
    print("causal targets (position, next token):", causal_targets)
    print("masked-LM target:", (masked_position, tokens[masked_position]), "with both left and right context")
    print("Causal prediction forbids future context; masked prediction hides selected tokens.")

# Causal language models predict a later token from a prefix, while masked models reconstruct deliberately hidden tokens.
if __name__ == "__main__":
    main()
