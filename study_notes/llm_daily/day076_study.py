"""Day 76: Study hallucination patterns and when prompts fail.

A dependency-light, local demonstration for the Day 76 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def classify(answer, evidence):
    # Empty output is treated as abstention before the simple lexical grounding check is attempted.
    if not answer: return "abstained"
    if answer.lower() in evidence.lower(): return "grounded"
    return "unsupported"

def main():
    # Each answer contrasts literal evidence overlap, an unsupported claim, and an explicit lack of answer.
    evidence = "The local note says attention uses keys."
    # Iteration applies the same deliberately simple evidence rule to grounded, unsupported, and empty responses.
    for answer in ["attention uses keys", "attention invented a database", ""]: print(repr(answer), "->", classify(answer, evidence))
    print("Grounding checks reduce one hallucination path but do not establish truth of the source.")

# Hallucination analysis asks whether a fluent answer is supported, not merely whether it sounds plausible.
if __name__ == "__main__":
    main()
