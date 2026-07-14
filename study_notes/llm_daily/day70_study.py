"""Day 70: Evaluate outputs and write down common failure cases.

A dependency-light, local demonstration for the Day 70 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def score(output, reference):
    if output == reference: return "correct"
    if not output: return "missing"
    return "wrong-or-unsupported"

def main():
    cases = [("Paris", "Paris"), ("Lyon", "Paris"), ("", "Tokyo")]
    buckets = [score(output, reference) for output, reference in cases]
    print("evaluation buckets:", buckets)
    print("A useful failure log retains input, expected output, actual output, and category.")

if __name__ == "__main__":
    main()
