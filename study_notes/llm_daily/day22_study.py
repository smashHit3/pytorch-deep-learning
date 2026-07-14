"""Day 22: Contrast order-sensitive text sequences with tabular feature rows."""


def next_token_pairs(tokens):
    return list(zip(tokens[:-1], tokens[1:]))


def main():
    tokens = "models learn ordered context".split()
    reordered = ["context", "ordered", "learn", "models"]
    tabular_row = {"age": 30, "income": 50000, "region": "west"}
    print("sequence pairs:", next_token_pairs(tokens))
    print("reordered sequence pairs:", next_token_pairs(reordered))
    print("tabular row (named columns retain meaning when displayed in another order):", tabular_row)
    print("Text models must preserve position because changing token order changes the target.")

if __name__ == "__main__":
    main()
