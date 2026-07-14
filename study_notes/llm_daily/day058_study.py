"""Day 58: Learn why dataset quality matters more than raw size in many cases.

A dependency-light, local demonstration for the Day 58 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def quality_flags(record):
    # The global seen set makes duplicate detection depend on records processed earlier in this local run.
    return {"duplicate": record in seen, "noisy": "###" in record, "relevant": "transformer" in record.lower()}

def main():
    global seen
    # Initialize state once so the repeated tutorial record is flagged only on its second encounter.
    seen = set()
    for record in ["Transformer attention tutorial", "Transformer attention tutorial", "### corrupted"]:
        print(record, "->", quality_flags(record)); seen.add(record)
    # The printed caveat matters because keyword flags cannot judge source quality, licensing, or nuanced relevance.
    print("Toy flags are triage signals, not a complete data-quality system.")

# Cleaner and more representative examples reduce the chance that extra data merely reinforces noise or bias.
if __name__ == "__main__":
    main()
