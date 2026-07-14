"""Day 107: Define scope, dataset, evaluation method, and demo goal.

A dependency-light, local demonstration for the Day 107 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import json

def main():
    brief = {"project": "local study-note QA", "scope": "fixed local notes", "metric": "grounded-answer rate", "demo": "query -> evidence -> answer"}
    query = "What does attention use?"; evidence = "Attention uses queries, keys, and values."
    answer = evidence if "attention" in query.lower() else "No local evidence."
    log = {"event": "demo", "query": query, "answer": answer, "success": answer != "No local evidence."}
    print("project brief:", json.dumps(brief)); print("structured log:", json.dumps(log)); print("health check: ok")

if __name__ == "__main__":
    main()
