"""Day 79: Generate deterministic sparse embeddings for local documents."""

import re


def embed(text, vocabulary):
    """Return a simple count-vector embedding to expose the embedding pipeline."""
    words = re.findall(r"[a-z]+", text.lower())
    return [words.count(word) for word in vocabulary]


def main():
    # Document values are tokenized together so every sparse vector shares the same ordered dimensions.
    documents = {
        "attention": "Attention combines queries keys and values",
        "retrieval": "Retrieval ranks local documents for a query",
        "safety": "Grounded answers cite supplied evidence",
    }
    # Sorting makes the count-vector coordinate order deterministic and inspectable in the printed vocabulary.
    vocabulary = sorted(set(re.findall(r"[a-z]+", " ".join(documents.values()).lower())))
    print("vocabulary dimensions:", len(vocabulary), vocabulary)
    # Each vector position is a word frequency, so matching terms create direct overlap for later retrieval.
    for name, text in documents.items():
        print(f"{name} embedding:", embed(text, vocabulary))
    print("These count vectors are transparent stand-ins for learned dense embeddings.")

# Sparse count vectors provide deterministic local embeddings, making their overlap-based scores easy to inspect.
if __name__ == "__main__":
    main()
