"""Day 77: Write a prompt engineering checklist for future projects.

A dependency-light, local demonstration for the Day 77 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    checklist = ["state task", "supply relevant context", "add representative examples", "set output schema", "define constraints", "evaluate failures"]
    for item in checklist: print("[ ]", item)
    print("A checklist improves repeatability; it does not replace task evaluation.")

if __name__ == "__main__":
    main()
