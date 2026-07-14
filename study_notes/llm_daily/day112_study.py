"""Day 112: Create a local learning retrospective and prioritized next steps."""


def summarize(milestones, evidence, gaps):
    """Build a concise retrospective from artifacts instead of a serving demo."""
    return {
        "learned": milestones,
        "evidence": evidence,
        "gaps": gaps,
        "next_steps": [
            "repeat an evaluation with a held-out test set",
            "read one Transformer implementation end to end",
            "expand the strongest project with measured reliability checks",
        ],
    }


def main():
    retrospective = summarize(
        milestones=["tensor and optimization basics", "causal attention and mini language models", "retrieval, evaluation, and deployment tradeoffs"],
        evidence=["ran local PyTorch exercises", "recorded toy-model limitations", "compared retrieval and safety policies"],
        gaps=["large-scale training", "real model fine-tuning", "production monitoring"],
    )
    print("final retrospective:")
    for section, items in retrospective.items():
        print(f"{section}:")
        for item in items:
            print(" -", item)
    print("Revise the gaps and next steps to match your own artifacts and career goals.")

if __name__ == "__main__":
    main()
