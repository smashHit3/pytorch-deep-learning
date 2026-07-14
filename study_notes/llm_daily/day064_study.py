"""Day 64: Contrast a base-model continuation prompt with an instruction format."""


def main():
    task = "The lesson was clear."
    base_prompt = f"Review: {task}\nSentiment:"
    instruction_prompt = f"Instruction: Classify the sentiment as positive or negative.\nInput: {task}\nResponse:"
    print("base model prompt (continues text):\n", base_prompt)
    print("instruction-tuned prompt (follows a requested task):\n", instruction_prompt)
    print("Both need evaluation; instruction tuning changes behavior, not the need for factual or safety checks.")

if __name__ == "__main__":
    main()
