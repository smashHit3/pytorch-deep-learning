"""Day 77: Write a prompt engineering checklist for future projects.

A dependency-light, local demonstration for the Day 77 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # The ordered checklist moves from task definition through context and constraints to failure evaluation.
    checklist = ["state task", "supply relevant context", "add representative examples", "set output schema", "define constraints", "evaluate failures"]
    # Bracket markers make this a reusable planning output rather than a claim that the steps are complete.
    # One printed line per item leaves a visible completion marker beside every planning requirement.
    for item in checklist: print("[ ]", item)
    print("A checklist improves repeatability; it does not replace task evaluation.")

# The execution guard ensures this reusable checklist is printed only when the study lesson is intentionally run.
if __name__ == "__main__":
    main()
