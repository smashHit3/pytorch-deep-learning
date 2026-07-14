"""Day 11: Study softmax, cross-entropy, and why classification uses them.

A dependency-light, local demonstration for the Day 11 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import math

def softmax(logits):
    # Subtracting the largest logit keeps exponentials numerically stable without changing probabilities.
    peak = max(logits)
    values = [math.exp(x - peak) for x in logits]
    return [x / sum(values) for x in values]

def main():
    # target is an index into the probability vector, so it selects the probability assigned to class 0.
    logits, target = [2.0, 1.0, -1.0], 0
    probabilities = softmax(logits)
    print("probabilities:", [round(x, 4) for x in probabilities])
    # Negative log likelihood grows sharply when the target class receives little probability mass.
    print("cross entropy:", round(-math.log(probabilities[target]), 4))

# Softmax turns logits into normalized probabilities; cross-entropy then penalizes probability assigned away from the target.
if __name__ == "__main__":
    main()
