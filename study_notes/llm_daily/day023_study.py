"""Day 23: Compare word, hand-split subword, and character tokenizations."""


def toy_subwords(word):
    """Show a deterministic stand-in for learned BPE/WordPiece segmentation."""
    return [word[:4], f"##{word[4:]}"] if len(word) > 4 else [word]


def main():
    # Whitespace splitting, the hand-written subword rule, and character extraction offer three granularities.
    text = "tokenization helps models"
    words = text.split()
    # The nested comprehension flattens the pieces for every word into one model input sequence.
    subwords = [piece for word in words for piece in toy_subwords(word)]
    # Removing spaces makes characters represent lexical content rather than whitespace separators.
    characters = list(text.replace(" ", ""))
    print("word tokens:", words)
    print("subword tokens:", subwords)
    print("character tokens:", characters)
    print("A production tokenizer learns its splits from data; this split only exposes the granularity tradeoff.")

# Tokenization trades vocabulary size against sequence length: smaller units cover more text but create longer inputs.
if __name__ == "__main__":
    main()
