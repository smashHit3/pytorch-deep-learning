"""Day 47: Learn greedy decoding, sampling, and temperature.

A dependency-light, local demonstration for the Day 47 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math, random

def sample(logits, temperature, seed=0):
    random.seed(seed); scaled = [x / temperature for x in logits]
    values = [math.exp(x-max(scaled)) for x in scaled]; total = sum(values)
    point, running = random.random(), 0.0
    for index, value in enumerate(values):
        running += value / total
        if point <= running: return index

def main():
    logits = [2.0, 1.0, 0.0]
    print("greedy:", max(range(3), key=logits.__getitem__))
    print("temperature .5 / 1.5:", sample(logits, .5), sample(logits, 1.5))

if __name__ == "__main__":
    main()
