"""Day 10: Learn probability basics: distributions, expectation, and variance.

A dependency-light, local demonstration for the Day 10 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import statistics

def main():
    samples = [1, 2, 2, 4, 6]
    print("mean:", statistics.fmean(samples), "population variance:", statistics.pvariance(samples))
    print("empirical P(x >= 4):", sum(x >= 4 for x in samples) / len(samples))

if __name__ == "__main__":
    main()
