"""Day 35: Review all Transformer submodules in one diagram.

A dependency-light, local demonstration for the Day 35 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    diagram = ["token IDs", "token + position embeddings", "[attention -> add & norm -> feed-forward -> add & norm] × layers", "linear vocabulary head", "next-token logits"]
    for index, node in enumerate(diagram): print("  " * min(index, 2) + "→ " + node)
    print("Each block preserves sequence length and embedding width.")

if __name__ == "__main__":
    main()
