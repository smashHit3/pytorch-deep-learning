"""Day 74: Measure prompt quality using consistency and error analysis.

A dependency-light, local demonstration for the Day 74 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

from collections import Counter

def main():
    repeated_outputs = ["A", "A", "B", "A", "B"]
    errors = ["format", "reasoning", "format"]
    print("output consistency:", Counter(repeated_outputs))
    print("error buckets:", Counter(errors))
    print("Count patterns before changing a prompt; one anecdote is weak evidence.")

if __name__ == "__main__":
    main()
