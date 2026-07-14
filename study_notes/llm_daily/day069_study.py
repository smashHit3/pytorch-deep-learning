"""Day 69: Fine-tune a small model for summarization or Q&A.

A dependency-light, local demonstration for the Day 69 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    examples = [("What is 2+2?", "4"), ("What color is grass?", "green")]
    baseline = {question: "I am not adapted" for question, _ in examples}
    adapted = dict(examples)
    for question, expected in examples: print(question, "| baseline:", baseline[question], "| adapted:", adapted[question], "| expected:", expected)

if __name__ == "__main__":
    main()
