"""Day 96: Build a compact human-evaluation sheet for project outputs."""


def make_sheet(prompt, answer):
    # Separate blank criteria let a reviewer score grounding, completeness, and clarity independently.
    return {
        "prompt": prompt,
        "answer": answer,
        "grounded (0-2)": "",
        "complete (0-2)": "",
        "clear (0-2)": "",
        "rationale": "",
        "reviewer": "",
    }


def main():
    # The sheet captures a concrete prompt-answer pair before reviewers add scores and written rationale.
    sheet = make_sheet("What does attention use?", "Attention uses queries, keys, and values.")
    print("human evaluation sheet:")
    # Iterating fields prints the review contract, including evidence for any score rather than a single verdict.
    for field, value in sheet.items():
        print(f"  {field}: {value}")
    print("Use independent reviewers, a written rubric, and a rationale for each score.")

# Human evaluation records criteria separately so a single overall impression does not hide correctness or safety failures.
if __name__ == "__main__":
    main()
