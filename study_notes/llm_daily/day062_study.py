"""Day 62: Read about distributed training at a high level.

A dependency-light, local demonstration for the Day 62 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    # Each inner list is one worker's gradient for the same two model parameters.
    worker_gradients = [[1.0, 3.0], [3.0, 5.0]]
    # zip aligns parameter coordinates before averaging, the local analogue of an all-reduce mean.
    averaged = [sum(column) / len(worker_gradients) for column in zip(*worker_gradients)]
    # The output shows that each averaged coordinate combines the same parameter coordinate from all workers.
    print("worker gradients:", worker_gradients, "all-reduce average:", averaged)
    print("More workers reduce per-worker data but add synchronization work.")

# Distributed training divides work across devices, but communication and synchronization limit ideal linear speedup.
if __name__ == "__main__":
    main()
