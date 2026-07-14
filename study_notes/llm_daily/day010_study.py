"""Day 10: Learn probability basics: distributions, expectation, and variance.

A dependency-light, local demonstration for the Day 10 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import statistics

def main():
    # Repeated values make this short list an empirical distribution rather than five unique outcomes.
    samples = [1, 2, 2, 4, 6]
    # pvariance divides by all five observations, treating this list as the complete population in the example.
    print("mean:", statistics.fmean(samples), "population variance:", statistics.pvariance(samples))
    # The Boolean sum counts observations meeting the event, then division converts it to a sample frequency.
    print("empirical P(x >= 4):", sum(x >= 4 for x in samples) / len(samples))

# Expectation summarizes a distribution's weighted center, while variance measures its spread around that center.
if __name__ == "__main__":
    main()
