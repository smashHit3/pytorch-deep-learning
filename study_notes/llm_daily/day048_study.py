"""Day 48: Generate short continuations with greedy and sampled decoding."""

import math
import random


def sample(logits, temperature, seed):
    random.seed(seed)
    scaled = [value / temperature for value in logits]
    weights = [math.exp(value - max(scaled)) for value in scaled]
    point, running = random.random(), 0.0
    for index, weight in enumerate(weights):
        running += weight / sum(weights)
        if point <= running:
            return index
    return len(logits) - 1


def generate(strategy, steps=5):
    vocabulary = ["the", "model", "learns"]
    sequence, current = ["the"], 0
    for step in range(steps):
        logits = [2.0, 1.0 + (current == 1), 0.4]
        current = max(range(3), key=logits.__getitem__) if strategy == "greedy" else sample(logits, 1.5, step)
        sequence.append(vocabulary[current])
    return " ".join(sequence)


def main():
    print("greedy:", generate("greedy"))
    print("sampled:", generate("sampled"))
    print("Greedy is deterministic; sampling can explore lower-probability continuations.")

if __name__ == "__main__":
    main()
