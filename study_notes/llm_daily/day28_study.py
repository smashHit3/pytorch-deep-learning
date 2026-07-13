"""Day 28: Write notes on why Transformers replaced recurrent models.

A dependency-light, local demonstration for the Day 28 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    comparison = {"RNN": ["sequential computation", "single compressed state"], "Transformer": ["parallel token processing", "direct attention paths"]}
    for architecture, traits in comparison.items(): print(architecture + ":", "; ".join(traits))
    print("Transformers still have costs: attention memory grows with sequence length.")

if __name__ == "__main__":
    main()
