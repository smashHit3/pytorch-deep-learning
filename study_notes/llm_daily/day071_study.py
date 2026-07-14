"""Day 71: Compare zero-shot, few-shot, and rationale-requesting prompts."""


def render(examples=(), request_rationale=False):
    prompt = "Classify sentiment as positive or negative.\n"
    for text, label in examples:
        prompt += f"Text: {text}\nLabel: {label}\n"
    if request_rationale:
        prompt += "Briefly explain the evidence before the label.\n"
    return prompt + "Text: The explanation was helpful.\n"


def main():
    print("zero-shot:\n", render() + "Label:")
    print("few-shot:\n", render([("I liked it", "positive"), ("It was poor", "negative")]) + "Label:")
    print("rationale-requesting:\n", render(request_rationale=True) + "Evidence:\nLabel:")
    print("Evaluate rationale prompting for accuracy and faithfulness; a requested explanation is not proof of reasoning.")

if __name__ == "__main__":
    main()
