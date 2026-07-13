"""Day 110: Add a structured log, a small evaluation, and usage instructions."""

import json


def run_query(query):
    answer = "Attention uses queries, keys, and values." if "attention" in query.lower() else "I do not have local evidence."
    return {"event": "answer", "query": query, "answer": answer, "grounded": answer != "I do not have local evidence."}


def main():
    cases = [("What does attention use?", True), ("What is the launch date?", False)]
    logs = [run_query(query) for query, _ in cases]
    accuracy = sum(log["grounded"] == expected for log, (_, expected) in zip(logs, cases)) / len(cases)
    print("usage: call run_query('What does attention use?') with a local study question.")
    for log in logs:
        print("structured log:", json.dumps(log, sort_keys=True))
    print("simple grounded/abstain evaluation:", accuracy)

if __name__ == "__main__":
    main()
