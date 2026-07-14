"""Day 41: Trace tensor shapes through the full block.

A dependency-light, local demonstration for the Day 41 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    batch, sequence, width, heads = 2, 5, 12, 3
    assert width % heads == 0
    print({"input": (batch, sequence, width), "QKV per head": (batch, heads, sequence, width // heads), "attention": (batch, heads, sequence, sequence), "output": (batch, sequence, width)})

if __name__ == "__main__":
    main()
