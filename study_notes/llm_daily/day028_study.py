"""Day 28: Write notes on why Transformers replaced recurrent models.

A dependency-light, local demonstration for the Day 28 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    # Each entry contrasts the path through time and the way earlier context is represented.
    comparison = {"RNN": ["sequential computation", "single compressed state"], "Transformer": ["parallel token processing", "direct attention paths"]}
    # Dictionary iteration emits one architecture and its paired tradeoff description on each pass.
    for architecture, traits in comparison.items(): print(architecture + ":", "; ".join(traits))
    # This cost caveat matters because full attention still has a pairwise token-memory burden.
    print("Transformers still have costs: attention memory grows with sequence length.")

# The guarded notes emphasize that attention can connect distant tokens without recurrent step-by-step propagation.
if __name__ == "__main__":
    main()
