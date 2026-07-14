"""Day 61: Study checkpoints, validation curves, and training instability.

A dependency-light, local demonstration for the Day 61 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    checkpoints = []
    validation_losses = [1.2, 0.9, 0.82, 1.4]
    for step, loss in enumerate(validation_losses, start=1):
        state = {"step": step, "validation_loss": loss, "weights": "toy-state"}
        checkpoints.append(state)
    best = min(checkpoints, key=lambda item: item["validation_loss"])
    print("best checkpoint:", best)
    print("Last loss rising from", validation_losses[-2], "to", validation_losses[-1], "is a signal to investigate, not proof of cause.")

if __name__ == "__main__":
    main()
