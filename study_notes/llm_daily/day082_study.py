"""Day 82: Build a minimal retrieval pipeline over local notes or documents.

A dependency-light, local demonstration for the Day 82 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math
import re

def vector(text, vocabulary):
    # Counts are aligned to a shared vocabulary, creating comparable sparse vectors for all texts.
    words = re.findall(r"[a-z]+", text.lower())
    return [words.count(word) for word in vocabulary]

def cosine(left, right):
    # Cosine normalizes the dot product by both vector lengths to rank overlap rather than raw word count.
    dot = sum(a*b for a,b in zip(left,right)); size = math.sqrt(sum(a*a for a in left))*math.sqrt(sum(b*b for b in right))
    return dot/size if size else 0.0

def main():
    # Only local document terms define dimensions, so unknown query terms cannot create retrieval evidence.
    documents = [("attention", "Attention combines queries, keys, and values."), ("retrieval", "Retrieval ranks local documents for a query."), ("safety", "Grounded answers should cite supplied evidence.")]
    vocabulary = sorted(set(re.findall(r"[a-z]+", " ".join(text for _, text in documents).lower())))
    query = "How does retrieval rank documents?"; query_vector = vector(query, vocabulary)
    # Ranking retains score, document name, and text so the selected context remains attributable.
    ranked = sorted(((cosine(query_vector, vector(text, vocabulary)), name, text) for name,text in documents), reverse=True)
    score, name, text = ranked[0]
    print("top local result:", name, "score:", round(score, 3))
    print("grounded context:", text)
    # A zero score produces abstention rather than treating the arbitrary top-ranked document as support.
    print("answer:", text if score else "I do not have supporting local evidence.")

# The toy retriever ranks count vectors by cosine similarity, then answers only from the highest-scoring local evidence.
if __name__ == "__main__":
    main()
