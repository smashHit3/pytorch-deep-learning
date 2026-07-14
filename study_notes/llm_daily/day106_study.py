"""Day 106: Choose a final LLM project using explicit constraints."""


def choose_project(options):
    """Pick the highest-scoring small, local, evaluable project option."""
    # Summing the fixed criteria makes selection deterministic, although real choices may require weighted tradeoffs.
    return max(options, key=lambda item: sum(item["scores"].values()))


def main():
    # Each option is scored on feasibility, evaluability, and demo value using the same three rubric dimensions.
    options = [
        {"name": "chatbot", "scores": {"local_scope": 1, "evaluation": 1, "demo_value": 2}},
        {"name": "summarizer", "scores": {"local_scope": 2, "evaluation": 2, "demo_value": 1}},
        {"name": "tutor", "scores": {"local_scope": 1, "evaluation": 1, "demo_value": 2}},
        {"name": "study-note QA", "scores": {"local_scope": 2, "evaluation": 2, "demo_value": 2}},
    ]
    # max returns the first highest total, so tied project candidates would need an additional explicit policy.
    choice = choose_project(options)
    print("candidates:", [option["name"] for option in options])
    print("chosen project:", choice["name"], "selection scores:", choice["scores"])
    print("Choose a project whose scope and evaluation can be completed with the available time and data.")

# Explicit constraints turn a broad project idea into a feasible choice that can be evaluated with available resources.
if __name__ == "__main__":
    main()
