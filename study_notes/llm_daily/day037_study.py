"""Day 37: Read the encoder-decoder overview in *Attention Is All You Need*.

A dependency-light, local demonstration for the Day 37 LLM roadmap topic.
It uses a toy-sized example and labels production-scale limitations.
"""

def main():
    # Uppercasing is only a visible stand-in for encoder representations, not a learned translation encoder.
    source = ["translate", "this", "text"]; encoder_memory = [word.upper() for word in source]
    # The beginning-of-sequence token initializes the decoder before it predicts its first target token.
    decoder_prefix = ["<BOS>"]
    # The printed memory is a toy sequence of encoder outputs that decoder cross-attention would consult.
    print("encoder memory:", encoder_memory)
    print("decoder prefix:", decoder_prefix, "cross-attends to encoder memory while generating the next target token")

# The direct-execution guard confines this architecture reading prompt to an intentional lesson run.
if __name__ == "__main__":
    main()
