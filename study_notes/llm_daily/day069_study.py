"""Day 69: Fine-tune a small model for summarization or Q&A.

A dependency-light, local demonstration for the Day 69 LLM roadmap topic.
It deliberately uses a toy-sized example: inspect its assumptions before transferring
any conclusion to a trained language model or production system.
"""

def main():
    # These question-answer pairs are deliberately small examples of the behavior a task adapter should learn.
    examples = [("What is 2+2?", "4"), ("What color is grass?", "green")]
    baseline = {question: "I am not adapted" for question, _ in examples}
    # The dictionary is a transparent idealized adaptation, not evidence that a learned model generalizes.
    adapted = dict(examples)
    # Each row compares both systems against the same expected answer for that question.
    for question, expected in examples: print(question, "| baseline:", baseline[question], "| adapted:", adapted[question], "| expected:", expected)

# Task-specific fine-tuning should be judged on held-out summaries or answers, not only its training loss.
if __name__ == "__main__":
    main()
