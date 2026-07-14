"""Day 104: Study API serving patterns and request limits.

A dependency-light, local demonstration for the Day 104 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

import json

def handle(request, count, limit=2):
    # Counts at or above the inclusive limit are rejected before prompt processing to protect shared capacity.
    if count >= limit: return {"status": 429, "error": "rate limit"}
    # Uppercasing is a transparent placeholder response that exposes the request/response contract without a model.
    prompt = request.get("prompt", "")
    return {"status": 200, "response": prompt.upper(), "request_id": count + 1}

def main():
    # Three sequential counts demonstrate two accepted requests followed by a deterministic rate-limit response.
    for count in range(3): print(json.dumps(handle({"prompt": "local request"}, count)))

# Request limits protect shared serving capacity by bounding the work one client can make the system perform.
if __name__ == "__main__":
    main()
