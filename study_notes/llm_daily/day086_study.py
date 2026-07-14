"""Day 86: Study chunking, metadata, and context packing.

A dependency-light, local demonstration for the Day 86 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def chunk(words, size=4, overlap=1):
    step = size - overlap
    return [words[i:i+size] for i in range(0, len(words), step) if words[i:i+size]]

def main():
    words = "metadata keeps source and page attached to each retrieval chunk".split()
    for index, piece in enumerate(chunk(words)): print({"chunk": index, "text": " ".join(piece), "source": "local-note", "page": 1})
    print("Packing must respect a context budget and preserve enough metadata for attribution.")

if __name__ == "__main__":
    main()
