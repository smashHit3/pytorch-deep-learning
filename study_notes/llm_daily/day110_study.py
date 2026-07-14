"""Day 110: Add a structured log, a small evaluation, and usage instructions."""

import json


def run_query(query):
    # This local policy returns evidence only for attention questions and otherwise reports an abstention message.
    answer = "Attention uses queries, keys, and values." if "attention" in query.lower() else "I do not have local evidence."
    return {"event": "answer", "query": query, "answer": answer, "grounded": answer != "I do not have local evidence."}


def main():
    # Expected Booleans label one supported and one unsupported case for a compact grounded/abstain evaluation.
    cases = [("What does attention use?", True), ("What is the launch date?", False)]
    logs = [run_query(query) for query, _ in cases]
    # Pairing logs with labels computes whether the policy chose grounded output or abstention as intended.
    accuracy = sum(log["grounded"] == expected for log, (_, expected) in zip(logs, cases)) / len(cases)
    print("usage: call run_query('What does attention use?') with a local study question.")
    for log in logs:
        print("structured log:", json.dumps(log, sort_keys=True))
    print("simple grounded/abstain evaluation:", accuracy)

# Structured logs connect inputs, decisions, and outcomes, making small evaluations reproducible and diagnosable.
if __name__ == "__main__":
    main()
