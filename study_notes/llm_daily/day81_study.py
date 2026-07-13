"""Day 81: Model a vector database record and its retrieval workflow in memory."""


def main():
    vector_store = [
        {"id": "note-1", "vector": [0.8, 0.1, 0.0], "metadata": {"source": "attention.md", "topic": "attention"}},
        {"id": "note-2", "vector": [0.1, 0.9, 0.2], "metadata": {"source": "retrieval.md", "topic": "retrieval"}},
    ]
    query_vector = [0.0, 1.0, 0.1]
    scored = sorted(
        ((sum(left * right for left, right in zip(query_vector, record["vector"])), record) for record in vector_store),
        reverse=True,
        key=lambda item: item[0],
    )
    print("workflow: embed document -> store vector + ID + metadata -> embed query -> nearest-neighbor search")
    for score, record in scored:
        print("score:", round(score, 3), "id:", record["id"], "metadata:", record["metadata"])
    print("A database adds persistence and indexing; this list only exposes the record contract.")

if __name__ == "__main__":
    main()
