"""Day 41: Trace tensor shapes through the full block.

A dependency-light, local demonstration for the Day 41 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    # width must divide by heads so every attention head has the same four-feature projection.
    batch, sequence, width, heads = 2, 5, 12, 3
    # This invariant prevents an invalid reshape when splitting Q, K, and V into heads.
    assert width % heads == 0
    # Attention's final two axes compare every one of the five queries against every one of the five keys.
    print({"input": (batch, sequence, width), "QKV per head": (batch, heads, sequence, width // heads), "attention": (batch, heads, sequence, sequence), "output": (batch, sequence, width)})

# Shape tracing verifies that batch and sequence axes survive each block while only feature dimensions are projected.
if __name__ == "__main__":
    main()
