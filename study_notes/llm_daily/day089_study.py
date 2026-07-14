"""Day 89: Add one tool or external function to your app.

A dependency-light, local demonstration for the Day 89 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def calculator(expression):
    import ast
    import operator

    # Only four arithmetic AST operator node types are mapped to callable Python operations.
    operations = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
    }

    def calculate(node):
        # The recursive evaluator accepts numeric constants and allowed binary expressions, rejecting every other node.
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](calculate(node.left), calculate(node.right))
        raise ValueError("unsupported expression")

    try:
        return {"result": calculate(ast.parse(expression, mode="eval").body)}
    except (SyntaxError, ValueError, ZeroDivisionError):
        return {"error": "unsupported expression"}

def main():
    # The second input demonstrates that a function-call AST is rejected rather than evaluated as Python code.
    print(calculator("(2 + 3) * 4"))
    print(calculator("open('secret')"))
    print("Production tools need stronger parsing, authorization, limits, and auditing.")

# Parsing an AST allowlist is safer than evaluating text directly because unsupported syntax is rejected before execution.
if __name__ == "__main__":
    main()
