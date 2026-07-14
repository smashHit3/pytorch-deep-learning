"""Day 72: Learn structured outputs and output constraints.

A dependency-light, local demonstration for the Day 72 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def validate(response):
    # The contract requires both a text answer and a list-valued sources field for downstream parsing.
    required = {"answer": str, "sources": list}
    return all(key in response and isinstance(response[key], kind) for key, kind in required.items())

def main():
    # The invalid record demonstrates that an answer alone does not satisfy the structured output schema.
    valid = {"answer": "Use local evidence.", "sources": ["note-1"]}
    invalid = {"answer": 42}
    # The two validation results contrast a complete type-correct record with one missing a required list.
    print("valid response:", validate(valid), "invalid response:", validate(invalid))
    print("Validate structure locally; schema validity does not guarantee factual correctness.")

# Structured constraints make downstream parsing more reliable by defining an output contract instead of free-form prose.
if __name__ == "__main__":
    main()
