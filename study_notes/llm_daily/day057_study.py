"""Day 57: Study causal LM versus masked LM objectives.

A dependency-light, local demonstration for the Day 57 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

import math

def main():
    tokens = ["the", "small", "model", "learns"]
    causal_targets = list(enumerate(tokens[1:], start=0))
    masked_position = 2
    print("causal targets (position, next token):", causal_targets)
    print("masked-LM target:", (masked_position, tokens[masked_position]), "with both left and right context")
    print("Causal prediction forbids future context; masked prediction hides selected tokens.")

if __name__ == "__main__":
    main()
