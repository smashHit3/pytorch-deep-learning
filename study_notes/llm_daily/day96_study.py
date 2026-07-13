"""Day 96: Build a compact human-evaluation sheet for project outputs."""


def make_sheet(prompt, answer):
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
    sheet = make_sheet("What does attention use?", "Attention uses queries, keys, and values.")
    print("human evaluation sheet:")
    for field, value in sheet.items():
        print(f"  {field}: {value}")
    print("Use independent reviewers, a written rubric, and a rationale for each score.")

if __name__ == "__main__":
    main()
