"""Day 50: Study a small implementation such as nanoGPT or minGPT.

A dependency-light, local demonstration for the Day 50 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    # Shapes track a batch of two five-token sequences through embeddings, four attention heads, and logits.
    shapes = {"token_ids": (2, 5), "embeddings": (2, 5, 16), "attention_scores": (2, 4, 5, 5), "logits": (2, 5, 20)}
    # The two length-five score axes correspond to query positions and key positions, respectively.
    # Formatting the labels to width 17 aligns the shape trace without altering the represented tensors.
    for name, shape in shapes.items(): print(f"{name:17} {shape}")
    print("Invariant: causal attention scores have one query and one key axis per head.")

# The shape listing makes explicit that each attention head carries one query and key axis for every token position.
if __name__ == "__main__":
    main()
