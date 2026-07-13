"""Day 98: Summarize alignment versus capability in simple language.

A dependency-light, local demonstration for the Day 98 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    cases = [
        {"output": "A confident answer", "capability": "answers quickly", "alignment": "unknown without evidence"},
        {"output": "I cannot verify this; here is the source", "capability": "may be less complete", "alignment": "grounded uncertainty"},
    ]
    for case in cases: print(case)
    print("Capability is what a model can do; alignment is whether its behavior follows intended values and constraints.")

if __name__ == "__main__":
    main()
