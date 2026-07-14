"""Day 70: Evaluate outputs and write down common failure cases.

A dependency-light, local demonstration for the Day 70 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def score(output, reference):
    # Exact match is checked before emptiness so the three labels form mutually exclusive evaluation buckets.
    if output == reference: return "correct"
    if not output: return "missing"
    return "wrong-or-unsupported"

def main():
    # The cases cover a matching answer, a conflicting answer, and an abstention-like empty response.
    cases = [("Paris", "Paris"), ("Lyon", "Paris"), ("", "Tokyo")]
    # Applying the classifier to every pair produces a parallel list whose order matches the original cases.
    buckets = [score(output, reference) for output, reference in cases]
    print("evaluation buckets:", buckets)
    print("A useful failure log retains input, expected output, actual output, and category.")

# Recording failure categories turns isolated bad outputs into concrete evaluation cases for the next iteration.
if __name__ == "__main__":
    main()
