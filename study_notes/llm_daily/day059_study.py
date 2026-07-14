"""Day 59: Study scaling laws and model/data/compute tradeoffs.

A dependency-light, local demonstration for the Day 59 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # Each pair allocates the same invented compute budget differently between model capacity and data volume.
    options = [(10, 90), (40, 60), (70, 30)]  # model units, data units; fixed compute budget
    # The minimum is intentionally a bottleneck proxy: extra model capacity cannot compensate for too little data.
    for model, data in options:
        effective = min(model * 1.2, data)  # deliberately invented proxy
        print({"model_units": model, "data_units": data, "toy_effective_capacity": effective})
    # This warning distinguishes the fabricated score from empirically fitted scaling-law measurements.
    print("Scaling laws are measured empirical relations; this is only a budget thought experiment.")

# Scaling tradeoffs matter because model size, data volume, and compute budget constrain one another in practice.
if __name__ == "__main__":
    main()
