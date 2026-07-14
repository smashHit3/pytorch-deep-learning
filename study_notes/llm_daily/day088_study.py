"""Day 88: Learn tool calling and function calling basics.

A dependency-light, local demonstration for the Day 88 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def add(left, right): return left + right

def dispatch(call):
    # The allowlist binds public tool names to trusted functions and their exact accepted argument keys.
    tools = {"add": (add, {"left", "right"})}
    name, arguments = call.get("name"), call.get("arguments", {})
    if name not in tools: return {"error": "unknown tool"}
    function, required = tools[name]
    # Reject missing, extra, or nonnumeric inputs before unpacking untrusted structured data into the function.
    if set(arguments) != required or not all(isinstance(value, (int, float)) for value in arguments.values()): return {"error": "invalid arguments"}
    return {"result": function(**arguments)}

def main():
    # The second request shows that a model-proposed name is not executed unless it is explicitly registered.
    print(dispatch({"name": "add", "arguments": {"left": 2, "right": 3}}))
    print(dispatch({"name": "shell", "arguments": {}}))
    print("An allowlist and argument validation are required before dispatching a tool.")

# Tool calling separates a model's structured request from trusted application code that validates and performs the action.
if __name__ == "__main__":
    main()
