"""Day 49: Write notes on why loss goes down but generation can still be bad.

A dependency-light, local demonstration for the Day 49 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    # Equal low losses can accompany different user-visible failures, so likelihood is not a complete quality score.
    observations = [(0.2, "repetitive but locally plausible"), (0.2, "unsupported continuation"), (1.0, "short but useful answer")]
    # Printing each pair keeps the lesson focused on the mismatch between scalar loss and qualitative behavior.
    for loss, text in observations: print({"loss": loss, "sample": text})
    # This interpretation separates teacher-forced likelihood from properties that emerge during decoding.
    print("Loss averages token probabilities on training-like data; generation also exposes decoding, coverage, and factuality problems.")

# The guarded reflection distinguishes optimization of token likelihood from the broader quality of generated text.
if __name__ == "__main__":
    main()
