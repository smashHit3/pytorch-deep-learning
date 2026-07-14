"""Day 12: Contrast learning rate, batch size, and epochs in a toy optimizer."""


def train(learning_rate, batch_size, epochs):
    weight = 0.0
    # Each scalar example pulls the single parameter toward its own value under squared-error loss.
    examples = [1.0, 2.0, 3.0, 4.0]
    updates = 0
    # Slicing by batch_size changes both gradient averaging and the number of updates per epoch.
    for _ in range(epochs):
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            gradient = sum(2 * (weight - value) for value in batch) / len(batch)
            weight -= learning_rate * gradient
            updates += 1
    return weight, updates


def main():
    # These configurations isolate how batch size and learning rate produce different update trajectories.
    for learning_rate, batch_size, epochs in ((0.1, 4, 3), (0.1, 1, 3), (0.3, 4, 3)):
        weight, updates = train(learning_rate, batch_size, epochs)
        print(
            f"lr={learning_rate}, batch_size={batch_size}, epochs={epochs} "
            f"-> weight={weight:.3f}, parameter updates={updates}"
        )

# Learning rate controls step size, while batch size and epochs control how much data each update and pass observe.
if __name__ == "__main__":
    main()
