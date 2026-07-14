"""Day 71: Compare zero-shot, few-shot, and rationale-requesting prompts."""


def render(examples=(), request_rationale=False):
    # Demonstrations serialize prior text-label pairs in the same format requested for the final input.
    prompt = "Classify sentiment as positive or negative.\n"
    for text, label in examples:
        prompt += f"Text: {text}\nLabel: {label}\n"
    # This optional instruction asks for evidence, but does not independently establish its faithfulness.
    if request_rationale:
        prompt += "Briefly explain the evidence before the label.\n"
    return prompt + "Text: The explanation was helpful.\n"


def main():
    # All three prompts classify the same sentence so output differences can be attributed to prompting choices.
    print("zero-shot:\n", render() + "Label:")
    print("few-shot:\n", render([("I liked it", "positive"), ("It was poor", "negative")]) + "Label:")
    print("rationale-requesting:\n", render(request_rationale=True) + "Evidence:\nLabel:")
    print("Evaluate rationale prompting for accuracy and faithfulness; a requested explanation is not proof of reasoning.")

# Demonstrations in a few-shot prompt constrain the expected pattern, but they also consume limited context space.
if __name__ == "__main__":
    main()
