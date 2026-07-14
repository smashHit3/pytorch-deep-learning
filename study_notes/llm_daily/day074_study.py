"""Day 74: Measure prompt quality using consistency and error analysis.

A dependency-light, local demonstration for the Day 74 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

from collections import Counter

def main():
    # Repeated samples from one prompt expose whether a nominally deterministic task varies across runs.
    repeated_outputs = ["A", "A", "B", "A", "B"]
    # Labels record failure types independently from the output values, enabling category-level aggregation.
    errors = ["format", "reasoning", "format"]
    print("output consistency:", Counter(repeated_outputs))
    # Counting categories makes recurring formatting failures distinguishable from rarer reasoning failures.
    print("error buckets:", Counter(errors))
    print("Count patterns before changing a prompt; one anecdote is weak evidence.")

# Consistency checks reveal prompt brittleness that a single appealing response could conceal.
if __name__ == "__main__":
    main()
