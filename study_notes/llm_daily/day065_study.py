"""Day 65: Study supervised fine-tuning datasets and formatting.

A dependency-light, local demonstration for the Day 65 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    record = {"instruction": "Classify the sentiment", "input": "The lesson was clear.", "response": "positive"}
    assert all(record.values())
    print("base-style:", record["input"] + " Sentiment:")
    print("instruction format:", record)
    print("Validation checks nonempty fields; real datasets also need provenance and review.")

if __name__ == "__main__":
    main()
