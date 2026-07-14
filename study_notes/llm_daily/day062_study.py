"""Day 62: Read about distributed training at a high level.

A dependency-light, local demonstration for the Day 62 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    worker_gradients = [[1.0, 3.0], [3.0, 5.0]]
    averaged = [sum(column) / len(worker_gradients) for column in zip(*worker_gradients)]
    print("worker gradients:", worker_gradients, "all-reduce average:", averaged)
    print("More workers reduce per-worker data but add synchronization work.")

if __name__ == "__main__":
    main()
