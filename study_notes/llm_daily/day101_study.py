"""Day 101: Learn quantization basics and memory tradeoffs.

A dependency-light, local demonstration for the Day 101 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    weights = [-1.2, -0.1, 0.4, 1.1]; scale = max(abs(x) for x in weights) / 127
    integers = [round(x/scale) for x in weights]; restored = [x*scale for x in integers]
    mae = sum(abs(a-b) for a,b in zip(weights,restored))/len(weights)
    print("int8-like values:", integers, "MAE:", round(mae, 6), "storage: roughly 1 byte/value vs 4 for float32")

if __name__ == "__main__":
    main()
