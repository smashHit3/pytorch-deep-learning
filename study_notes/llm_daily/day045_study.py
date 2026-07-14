"""Day 45: Build a vocabulary or simple tokenizer.

A dependency-light, local demonstration for the Day 45 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

from collections import Counter

def main():
    # This explicit mapping is the tokenizer contract: every observed token has one integer model input ID.
    text = "tiny models learn from tiny local corpora"
    tokens = text.split()
    vocab = {token: i for i, token in enumerate(sorted(set(tokens)))}
    # Each list element is looked up independently, so repeated words produce repeated input IDs.
    encoded = [vocab[t] for t in tokens]
    print("vocabulary:", vocab)
    # Pairing offset encoded lists makes visible the labels that a causal language model would predict.
    print("encoded:", encoded, "shifted pairs:", list(zip(encoded[:-1], encoded[1:])))

# A vocabulary establishes the reversible mapping between text units and the numeric IDs a model consumes.
if __name__ == "__main__":
    main()
