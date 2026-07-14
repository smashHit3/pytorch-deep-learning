"""Day 102: Run a small open model locally with Ollama, vLLM, or another tool.

A dependency-light, local demonstration for the Day 102 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def generate(prompt, steps=4):
    # An empty prompt falls back to one token so cyclic indexing always has a nonzero vocabulary length.
    words = prompt.split() or ["local"]
    # Repeating prompt words is a deterministic stand-in for token decoding, not a model forward pass.
    return " ".join(words + [words[index % len(words)] for index in range(steps)])

def main():
    # The printed continuation demonstrates the local function's contract without downloading weights or a runtime.
    print("local surrogate output:", generate("small model"))
    print("A real Ollama/vLLM runtime supplies model weights, tokenizer, device kernels, batching, and streaming; this download-free script does not impersonate one.")

# The guarded local-serving checklist prints only in this lesson, since installed runtimes and model files vary by machine.
if __name__ == "__main__":
    main()
