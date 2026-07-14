"""Day 89: Add one tool or external function to your app.

A dependency-light, local demonstration for the Day 89 LLM roadmap topic.
It uses a toy-sized example and explicitly labels where real systems need more rigor.
"""

def calculator(expression):
    import ast
    import operator

    operations = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
    }

    def calculate(node):
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
    print(calculator("(2 + 3) * 4"))
    print(calculator("open('secret')"))
    print("Production tools need stronger parsing, authorization, limits, and auditing.")

if __name__ == "__main__":
    main()
