"""Day 22: Contrast order-sensitive text sequences with tabular feature rows."""


def next_token_pairs(tokens):
    # Offset slices pair each token with its immediate successor, preserving sequence direction.
    return list(zip(tokens[:-1], tokens[1:]))


def main():
    # Reordering the same words changes their adjacent pairs, unlike named values in a tabular record.
    tokens = "models learn ordered context".split()
    reordered = ["context", "ordered", "learn", "models"]
    tabular_row = {"age": 30, "income": 50000, "region": "west"}
    print("sequence pairs:", next_token_pairs(tokens))
    print("reordered sequence pairs:", next_token_pairs(reordered))
    print("tabular row (named columns retain meaning when displayed in another order):", tabular_row)
    # This output contrasts ordered next-token targets with independently named tabular fields.
    print("Text models must preserve position because changing token order changes the target.")

# Sequence position carries meaning in text, whereas a tabular row normally treats feature columns as fixed fields.
if __name__ == "__main__":
    main()
