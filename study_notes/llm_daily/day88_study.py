"""Day 88: Learn tool calling and function calling basics.

A dependency-light, local demonstration for the Day 88 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def add(left, right): return left + right

def dispatch(call):
    tools = {"add": (add, {"left", "right"})}
    name, arguments = call.get("name"), call.get("arguments", {})
    if name not in tools: return {"error": "unknown tool"}
    function, required = tools[name]
    if set(arguments) != required or not all(isinstance(value, (int, float)) for value in arguments.values()): return {"error": "invalid arguments"}
    return {"result": function(**arguments)}

def main():
    print(dispatch({"name": "add", "arguments": {"left": 2, "right": 3}}))
    print(dispatch({"name": "shell", "arguments": {}}))
    print("An allowlist and argument validation are required before dispatching a tool.")

if __name__ == "__main__":
    main()
