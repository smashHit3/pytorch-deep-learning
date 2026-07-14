"""Day 91: Write notes on when to use plain prompting, RAG, or tools.

A dependency-light, local demonstration for the Day 91 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def choose(needs_external_facts, needs_action):
    # External action takes precedence because tools can both obtain facts and perform an operation.
    return "tool" if needs_action else "RAG" if needs_external_facts else "plain prompt"

def main():
    # The three Boolean combinations exercise no dependency, knowledge-only, and action-required task routes.
    scenarios = [(False, False), (True, False), (False, True)]
    # Each scenario passes facts and action flags independently so the precedence rule is observable in output.
    for facts, action in scenarios: print({"external_facts": facts, "action": action, "choice": choose(facts, action)})
    print("Choose the smallest mechanism that satisfies freshness, reliability, and cost needs.")

# Choosing prompting, retrieval, or tools depends on whether the task needs knowledge, external action, or neither.
if __name__ == "__main__":
    main()
