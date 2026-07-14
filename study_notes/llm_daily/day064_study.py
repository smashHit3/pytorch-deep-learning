"""Day 64: Contrast a base-model continuation prompt with an instruction format."""


def main():
    # The same sentence isolates the effect of prompt framing from changes in task input.
    task = "The lesson was clear."
    base_prompt = f"Review: {task}\nSentiment:"
    # Explicit labels delimit instruction, input, and desired response location for an instruction-following model.
    instruction_prompt = f"Instruction: Classify the sentiment as positive or negative.\nInput: {task}\nResponse:"
    # Newlines make the formatting boundaries visible to both the reader and the prompted model.
    print("base model prompt (continues text):\n", base_prompt)
    print("instruction-tuned prompt (follows a requested task):\n", instruction_prompt)
    print("Both need evaluation; instruction tuning changes behavior, not the need for factual or safety checks.")

# Instruction formatting supplies explicit task boundaries that a plain continuation prompt leaves implicit.
if __name__ == "__main__":
    main()
