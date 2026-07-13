"""Day 72: Learn structured outputs and output constraints.

A dependency-light, local demonstration for the Day 72 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def validate(response):
    required = {"answer": str, "sources": list}
    return all(key in response and isinstance(response[key], kind) for key, kind in required.items())

def main():
    valid = {"answer": "Use local evidence.", "sources": ["note-1"]}
    invalid = {"answer": 42}
    print("valid response:", validate(valid), "invalid response:", validate(invalid))
    print("Validate structure locally; schema validity does not guarantee factual correctness.")

if __name__ == "__main__":
    main()
