"""Day 91: Write notes on when to use plain prompting, RAG, or tools.

A dependency-light, local demonstration for the Day 91 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def choose(needs_external_facts, needs_action):
    return "tool" if needs_action else "RAG" if needs_external_facts else "plain prompt"

def main():
    scenarios = [(False, False), (True, False), (False, True)]
    for facts, action in scenarios: print({"external_facts": facts, "action": action, "choice": choose(facts, action)})
    print("Choose the smallest mechanism that satisfies freshness, reliability, and cost needs.")

if __name__ == "__main__":
    main()
