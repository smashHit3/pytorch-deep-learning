"""Day 98: Summarize alignment versus capability in simple language.

A dependency-light, local demonstration for the Day 98 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # The cases contrast a capable fast answer with a source-aware answer that exposes uncertainty.
    cases = [
        {"output": "A confident answer", "capability": "answers quickly", "alignment": "unknown without evidence"},
        {"output": "I cannot verify this; here is the source", "capability": "may be less complete", "alignment": "grounded uncertainty"},
    ]
    # Showing both labels together prevents behavior that appears useful from being mistaken for aligned behavior.
    # Each dictionary keeps the raw output beside separate capability and alignment interpretations.
    for case in cases: print(case)
    print("Capability is what a model can do; alignment is whether its behavior follows intended values and constraints.")

# Capability concerns what a model can do; alignment concerns whether it reliably does what people intend.
if __name__ == "__main__":
    main()
