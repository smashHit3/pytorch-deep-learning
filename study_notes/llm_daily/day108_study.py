"""Day 108: Build the first end-to-end version.

A dependency-light, local demonstration for the Day 108 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import json

def main():
    # The shared brief documents the first version's deliberately fixed corpus and measurable grounded-answer goal.
    brief = {"project": "local study-note QA", "scope": "fixed local notes", "metric": "grounded-answer rate", "demo": "query -> evidence -> answer"}
    # The answer is constrained to the available evidence; any other topic follows the explicit abstention branch.
    query = "What does attention use?"; evidence = "Attention uses queries, keys, and values."
    answer = evidence if "attention" in query.lower() else "No local evidence."
    # Recording the exact input and output verifies the complete query-to-evidence-to-answer path end to end.
    log = {"event": "demo", "query": query, "answer": answer, "success": answer != "No local evidence."}
    print("project brief:", json.dumps(brief)); print("structured log:", json.dumps(log)); print("health check: ok")

# An end-to-end first version validates the full data-to-output path before effort is spent on isolated refinements.
if __name__ == "__main__":
    main()
