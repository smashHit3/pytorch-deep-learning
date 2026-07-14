"""Day 58: Learn why dataset quality matters more than raw size in many cases.

A dependency-light, local demonstration for the Day 58 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def quality_flags(record):
    return {"duplicate": record in seen, "noisy": "###" in record, "relevant": "transformer" in record.lower()}

def main():
    global seen
    seen = set()
    for record in ["Transformer attention tutorial", "Transformer attention tutorial", "### corrupted"]:
        print(record, "->", quality_flags(record)); seen.add(record)
    print("Toy flags are triage signals, not a complete data-quality system.")

if __name__ == "__main__":
    main()
