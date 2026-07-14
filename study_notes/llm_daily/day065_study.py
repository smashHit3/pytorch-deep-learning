"""Day 65: Study supervised fine-tuning datasets and formatting.

A dependency-light, local demonstration for the Day 65 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    # A supervised record binds one instruction and input to the response behavior the model should imitate.
    record = {"instruction": "Classify the sentiment", "input": "The lesson was clear.", "response": "positive"}
    # Nonempty fields are only a structural baseline; they do not verify label correctness or data provenance.
    assert all(record.values())
    # The first format leaves the task implicit in continuation text, unlike the explicit structured record.
    print("base-style:", record["input"] + " Sentiment:")
    print("instruction format:", record)
    print("Validation checks nonempty fields; real datasets also need provenance and review.")

# Fine-tuning examples must pair the intended instruction format with the desired response so the model learns the contract.
if __name__ == "__main__":
    main()
