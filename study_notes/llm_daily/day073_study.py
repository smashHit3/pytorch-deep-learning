"""Day 73: Build a small prompt comparison set for one task.

A dependency-light, local demonstration for the Day 73 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # Both rows hold the task control constant while varying the prompt construction being compared.
    rows = [
        {"prompt": "direct", "control": "same task", "output": "positive", "expected": "positive"},
        {"prompt": "two examples", "control": "same task", "output": "negative", "expected": "positive"},
    ]
    # The Boolean makes every expected-output comparison explicit instead of relying on subjective inspection.
    # A false value identifies a prompt row requiring investigation under the shared task control.
    for row in rows: print(row, "pass=", row["output"] == row["expected"])
    print("Keep task, decoding, and evaluation examples controlled while comparing prompts.")

# A fixed prompt comparison set keeps task examples constant, making changes in output easier to attribute to the prompt.
if __name__ == "__main__":
    main()
