"""Day 50: Study a small implementation such as nanoGPT or minGPT.

A dependency-light, local demonstration for the Day 50 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    shapes = {"token_ids": (2, 5), "embeddings": (2, 5, 16), "attention_scores": (2, 4, 5, 5), "logits": (2, 5, 20)}
    for name, shape in shapes.items(): print(f"{name:17} {shape}")
    print("Invariant: causal attention scores have one query and one key axis per head.")

if __name__ == "__main__":
    main()
