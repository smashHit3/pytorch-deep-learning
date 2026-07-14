"""Day 90: Study simple agent loops and orchestration patterns.

A dependency-light, local demonstration for the Day 90 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    # State records the original question, tool observations, and a bounded step counter for inspection.
    state = {"query": "2 + 2", "observations": [], "steps": 0}
    # The two-step limit ensures the first pass observes a tool result and the second converts it to an answer.
    while state["steps"] < 2:
        if state["steps"] == 0: state["observations"].append("calculator returned 4")
        else: state["answer"] = state["observations"][-1]
        state["steps"] += 1
    # Printing complete state makes the intermediate observation and terminating condition auditable.
    print(state)
    print("Bounded steps and explicit stop conditions keep simple agent loops inspectable.")

# An agent loop alternates observation and action, so explicit stopping rules are needed to prevent unbounded work.
if __name__ == "__main__":
    main()
