"""Day 78: Learn what embeddings are and how semantic similarity works.

A dependency-light, local demonstration for the Day 78 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math
import re

def vector(text, vocabulary):
    # The vector has one count per vocabulary term, making its dimension consistent for queries and documents.
    words = re.findall(r"[a-z]+", text.lower())
    return [words.count(word) for word in vocabulary]

def cosine(left, right):
    # Dividing the dot product by both lengths compares overlap direction instead of raw document length.
    dot = sum(a*b for a,b in zip(left,right)); size = math.sqrt(sum(a*a for a in left))*math.sqrt(sum(b*b for b in right))
    return dot/size if size else 0.0

def main():
    # The vocabulary is built from local documents, so unknown query words contribute no count-vector evidence.
    documents = [("attention", "Attention combines queries, keys, and values."), ("retrieval", "Retrieval ranks local documents for a query."), ("safety", "Grounded answers should cite supplied evidence.")]
    vocabulary = sorted(set(re.findall(r"[a-z]+", " ".join(text for _, text in documents).lower())))
    query = "How does retrieval rank documents?"; query_vector = vector(query, vocabulary)
    # Ranking pairs each document with cosine similarity, then selects context only from the top local result.
    ranked = sorted(((cosine(query_vector, vector(text, vocabulary)), name, text) for name,text in documents), reverse=True)
    score, name, text = ranked[0]
    print("top local result:", name, "score:", round(score, 3))
    print("grounded context:", text)
    print("answer:", text if score else "I do not have supporting local evidence.")

# Embedding similarity compares vector direction, so cosine similarity can ignore differences in vector magnitude.
if __name__ == "__main__":
    main()
