"""Day 60: Learn learning rate warmup, decay, and gradient clipping.

A dependency-light, local demonstration for the Day 60 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def lr(step, warmup=3, total=10):
    return step / warmup if step < warmup else max(0.0, (total-step)/(total-warmup))

def main():
    gradients = [3.0, 4.0]; norm = sum(x*x for x in gradients) ** .5
    scale = min(1.0, 2.0 / norm)
    print("schedule:", [round(lr(s), 2) for s in range(10)])
    print("clipped gradient:", [round(x*scale, 2) for x in gradients])

if __name__ == "__main__":
    main()
