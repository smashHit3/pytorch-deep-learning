"""Day 99: Learn the inference pipeline from tokens to generated text.

A dependency-light, local demonstration for the Day 99 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    # Token IDs represent the already-tokenized prefix; logits provide one unnormalized score per next-ID choice.
    token_ids = [3, 7, 2]; logits = [0.1, 2.0, 0.4]
    # argmax is greedy decoding, selecting the largest score without temperature or stochastic sampling.
    # The key function looks up each candidate index's logit, returning index one for the maximum score 2.0.
    next_id = max(range(len(logits)), key=logits.__getitem__)
    print("input token IDs:", token_ids, "-> forward-pass logits:", logits, "-> decoded next ID:", next_id)
    print("A real pipeline also tokenizes text, repeatedly samples IDs, and detokenizes the final sequence.")

# Generation repeatedly feeds chosen token IDs back into the model, extending the sequence one token at a time.
if __name__ == "__main__":
    main()
