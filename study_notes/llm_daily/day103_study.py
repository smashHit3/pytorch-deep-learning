"""Day 103: Compare CPU versus GPU serving tradeoffs.

A dependency-light, local demonstration for the Day 103 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # These profiles are qualitative planning assumptions, not benchmark measurements for any particular hardware.
    profiles = {"CPU": {"startup": "simple", "parallel_tokens": 1, "memory": "system RAM"}, "GPU": {"startup": "device setup", "parallel_tokens": 64, "memory": "VRAM"}}
    # Presenting the same fields for both devices makes tradeoffs in parallelism and memory placement comparable.
    # Dictionary iteration prints one device profile at a time while retaining its associated hardware assumptions.
    for name, profile in profiles.items(): print(name, profile)
    print("Actual choices depend on model size, traffic, hardware availability, power, and measured latency.")

# GPUs favor parallel token computation, while CPUs can be simpler and cheaper for small or low-throughput workloads.
if __name__ == "__main__":
    main()
