"""Day 45: Build a vocabulary or simple tokenizer.

A dependency-light, local demonstration for the Day 45 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

from collections import Counter

def main():
    text = "tiny models learn from tiny local corpora"
    tokens = text.split()
    vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}
    encoded = [vocab[t] for t in tokens]
    print("vocabulary:", vocab)
    print("encoded:", encoded, "shifted pairs:", list(zip(encoded[:-1], encoded[1:])))

if __name__ == "__main__":
    main()
