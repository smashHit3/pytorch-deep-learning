"""Day 30: Learn query, key, and value intuition with a tiny example.

A dependency-light, local demonstration for the Day 30 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

import torch

def main():
    # The query and keys share width two, so their dot products produce one relevance score per key.
    query = torch.tensor([1., 0.]); keys = torch.tensor([[1., 0.], [0., 1.]])
    # Value rows need not resemble keys; they contain the content mixed according to attention weights.
    values = torch.tensor([[10., 0.], [0., 20.]])
    # Softmax normalizes scores across keys, and the resulting weights aggregate aligned value rows.
    weights = torch.softmax(keys @ query, dim=0); read = weights @ values
    print("query-key scores:", (keys @ query).tolist(), "weights:", weights.tolist())
    print("weighted value read:", read.tolist())

# Query-key scores decide where to look; the resulting weights mix the corresponding value vectors.
if __name__ == "__main__":
    main()
