"""Day 90: Study simple agent loops and orchestration patterns.

A dependency-light, local demonstration for the Day 90 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def main():
    state = {"query": "2 + 2", "observations": [], "steps": 0}
    while state["steps"] < 2:
        if state["steps"] == 0: state["observations"].append("calculator returned 4")
        else: state["answer"] = state["observations"][-1]
        state["steps"] += 1
    print(state)
    print("Bounded steps and explicit stop conditions keep simple agent loops inspectable.")

if __name__ == "__main__":
    main()
