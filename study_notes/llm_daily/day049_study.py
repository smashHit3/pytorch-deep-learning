"""Day 49: Write notes on why loss goes down but generation can still be bad.

A dependency-light, local demonstration for the Day 49 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    observations = [(0.2, "repetitive but locally plausible"), (0.2, "unsupported continuation"), (1.0, "short but useful answer")]
    for loss, text in observations: print({"loss": loss, "sample": text})
    print("Loss averages token probabilities on training-like data; generation also exposes decoding, coverage, and factuality problems.")

if __name__ == "__main__":
    main()
