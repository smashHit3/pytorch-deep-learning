"""Day 97: Compare several outputs and rank them with clear criteria.

A dependency-light, local demonstration for the Day 97 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    # Weights state the relative importance of groundedness, completeness, and clarity before scores are combined.
    rubric = {"grounded": 0.5, "complete": 0.3, "clear": 0.2}
    candidates = {"A": {"grounded": 2, "complete": 1, "clear": 2}, "B": {"grounded": 1, "complete": 2, "clear": 2}}
    # Each weighted sum applies the same rubric to every candidate, making the numerical ranking reproducible.
    # Comprehension keys preserve candidate names so a score remains traceable to its original output.
    scores = {name: sum(rubric[key]*value for key,value in row.items()) for name,row in candidates.items()}
    print("rubric:", rubric, "weighted rankings:", scores)
    print("Record a rationale and ties; a toy rubric is not a substitute for human review.")

# Pairwise rankings force criteria into comparative decisions, exposing tradeoffs that independent scores can obscure.
if __name__ == "__main__":
    main()
