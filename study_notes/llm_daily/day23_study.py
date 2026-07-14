"""Day 23: Compare word, hand-split subword, and character tokenizations."""


def toy_subwords(word):
    """Show a deterministic stand-in for learned BPE/WordPiece segmentation."""
    return [word[:4], f"##{word[4:]}"] if len(word) > 4 else [word]


def main():
    text = "tokenization helps models"
    words = text.split()
    subwords = [piece for word in words for piece in toy_subwords(word)]
    characters = list(text.replace(" ", ""))
    print("word tokens:", words)
    print("subword tokens:", subwords)
    print("character tokens:", characters)
    print("A production tokenizer learns its splits from data; this split only exposes the granularity tradeoff.")

if __name__ == "__main__":
    main()
