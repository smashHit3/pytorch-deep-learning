"""Day 103: Compare CPU versus GPU serving tradeoffs.

A dependency-light, local demonstration for the Day 103 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    profiles = {"CPU": {"startup": "simple", "parallel_tokens": 1, "memory": "system RAM"}, "GPU": {"startup": "device setup", "parallel_tokens": 64, "memory": "VRAM"}}
    for name, profile in profiles.items(): print(name, profile)
    print("Actual choices depend on model size, traffic, hardware availability, power, and measured latency.")

if __name__ == "__main__":
    main()
