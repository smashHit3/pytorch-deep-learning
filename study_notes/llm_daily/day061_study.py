"""Day 61: Study checkpoints, validation curves, and training instability.

A dependency-light, local demonstration for the Day 61 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    checkpoints = []
    # Each value stands in for a held-out loss measured after a successive training checkpoint.
    validation_losses = [1.2, 0.9, 0.82, 1.4]
    for step, loss in enumerate(validation_losses, start=1):
        state = {"step": step, "validation_loss": loss, "weights": "toy-state"}
        # Appending preserves a separate recoverable state record for each measured validation point.
        checkpoints.append(state)
    # Selecting the minimum validation loss chooses a saved state rather than assuming the last state is best.
    best = min(checkpoints, key=lambda item: item["validation_loss"])
    print("best checkpoint:", best)
    print("Last loss rising from", validation_losses[-2], "to", validation_losses[-1], "is a signal to investigate, not proof of cause.")

# Checkpoints preserve recoverable training state, while validation curves distinguish progress from instability or overfit.
if __name__ == "__main__":
    main()
