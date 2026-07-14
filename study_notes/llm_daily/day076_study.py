"""Day 76: Study hallucination patterns and when prompts fail.

A dependency-light, local demonstration for the Day 76 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def classify(answer, evidence):
    if not answer: return "abstained"
    if answer.lower() in evidence.lower(): return "grounded"
    return "unsupported"

def main():
    evidence = "The local note says attention uses keys."
    for answer in ["attention uses keys", "attention invented a database", ""]: print(repr(answer), "->", classify(answer, evidence))
    print("Grounding checks reduce one hallucination path but do not establish truth of the source.")

if __name__ == "__main__":
    main()
