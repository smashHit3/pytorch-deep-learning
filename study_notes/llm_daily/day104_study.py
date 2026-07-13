"""Day 104: Study API serving patterns and request limits.

A dependency-light, local demonstration for the Day 104 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import json

def handle(request, count, limit=2):
    if count >= limit: return {"status": 429, "error": "rate limit"}
    prompt = request.get("prompt", "")
    return {"status": 200, "response": prompt.upper(), "request_id": count + 1}

def main():
    for count in range(3): print(json.dumps(handle({"prompt": "local request"}, count)))

if __name__ == "__main__":
    main()
