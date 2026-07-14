"""Day 107: Define scope, dataset, evaluation method, and demo goal.

A dependency-light, local demonstration for the Day 107 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import json

def main():
    # The brief fixes the evidence scope and metric before the demo is run, making the project boundary testable.
    brief = {"project": "local study-note QA", "scope": "fixed local notes", "metric": "grounded-answer rate", "demo": "query -> evidence -> answer"}
    # This branch abstains unless the query names the supported topic, a simple grounded-answer policy.
    query = "What does attention use?"; evidence = "Attention uses queries, keys, and values."
    answer = evidence if "attention" in query.lower() else "No local evidence."
    # The structured log preserves the input, decision, and Boolean outcome for reproducible demonstration checks.
    log = {"event": "demo", "query": query, "answer": answer, "success": answer != "No local evidence."}
    print("project brief:", json.dumps(brief)); print("structured log:", json.dumps(log)); print("health check: ok")

# Defining data, metrics, and a demo goal up front makes project scope testable rather than an open-ended aspiration.
if __name__ == "__main__":
    main()
