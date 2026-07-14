"""Day 79: Generate deterministic sparse embeddings for local documents."""

import re


def embed(text, vocabulary):
    """Return a simple count-vector embedding to expose the embedding pipeline."""
    words = re.findall(r"[a-z]+", text.lower())
    return [words.count(word) for word in vocabulary]


def main():
    documents = {
        "attention": "Attention combines queries keys and values",
        "retrieval": "Retrieval ranks local documents for a query",
        "safety": "Grounded answers cite supplied evidence",
    }
    vocabulary = sorted(set(re.findall(r"[a-z]+", " ".join(documents.values()).lower())))
    print("vocabulary dimensions:", len(vocabulary), vocabulary)
    for name, text in documents.items():
        print(f"{name} embedding:", embed(text, vocabulary))
    print("These count vectors are transparent stand-ins for learned dense embeddings.")

if __name__ == "__main__":
    main()
