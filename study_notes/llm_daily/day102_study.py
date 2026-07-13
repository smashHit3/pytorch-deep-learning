"""Day 102: Run a small open model locally with Ollama, vLLM, or another tool.

A dependency-light, local demonstration for the Day 102 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def generate(prompt, steps=4):
    words = prompt.split() or ["local"]
    return " ".join(words + [words[index % len(words)] for index in range(steps)])

def main():
    print("local surrogate output:", generate("small model"))
    print("A real Ollama/vLLM runtime supplies model weights, tokenizer, device kernels, batching, and streaming; this download-free script does not impersonate one.")

if __name__ == "__main__":
    main()
