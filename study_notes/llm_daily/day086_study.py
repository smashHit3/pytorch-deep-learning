"""Day 86: Study chunking, metadata, and context packing.

A dependency-light, local demonstration for the Day 86 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def chunk(words, size=4, overlap=1):
    # Advancing by size minus overlap repeats boundary words, preserving some cross-chunk context.
    step = size - overlap
    return [words[i:i+size] for i in range(0, len(words), step) if words[i:i+size]]

def main():
    # Splitting the local sentence makes chunk contents small enough to rank while retaining source and page fields.
    words = "metadata keeps source and page attached to each retrieval chunk".split()
    # The printed records illustrate that retrieved content needs attribution metadata for later citation.
    for index, piece in enumerate(chunk(words)): print({"chunk": index, "text": " ".join(piece), "source": "local-note", "page": 1})
    print("Packing must respect a context budget and preserve enough metadata for attribution.")

# Chunk boundaries and metadata affect what can be retrieved, while context packing must fit the model's input budget.
if __name__ == "__main__":
    main()
