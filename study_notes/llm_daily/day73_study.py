"""Day 73: Build a small prompt comparison set for one task.

A dependency-light, local demonstration for the Day 73 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    rows = [
        {"prompt": "direct", "control": "same task", "output": "positive", "expected": "positive"},
        {"prompt": "two examples", "control": "same task", "output": "negative", "expected": "positive"},
    ]
    for row in rows: print(row, "pass=", row["output"] == row["expected"])
    print("Keep task, decoding, and evaluation examples controlled while comparing prompts.")

if __name__ == "__main__":
    main()
