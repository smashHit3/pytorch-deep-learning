"""Day 37: Read the encoder-decoder overview in *Attention Is All You Need*.

A dependency-light, local demonstration for the Day 37 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    source = ["translate", "this", "text"]; encoder_memory = [word.upper() for word in source]
    decoder_prefix = ["<BOS>"]
    print("encoder memory:", encoder_memory)
    print("decoder prefix:", decoder_prefix, "cross-attends to encoder memory while generating the next target token")

if __name__ == "__main__":
    main()
