#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NLP Inference Framework
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from argparse import ArgumentParser

from nlp_sources.data_processor import text_data
from nlp_sources.models import rnn, transformer


def parse_args():
    parser = ArgumentParser(description="NLP Text Classification Inference")
    
    parser.add_argument("--model", type=str, required=True,
                        choices=[rnn.MODEL_TYPE_LSTM, rnn.MODEL_TYPE_GRU, 
                                 transformer.MODEL_TYPE_TRANSFORMER],
                        help="Select model type")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--vocab-path", type=Path,
                        help="Path to vocabulary file")
    parser.add_argument("--text", type=str,
                        help="Text to classify")
    parser.add_argument("--text-file", type=Path,
                        help="File containing text to classify")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force use CPU")
    
    return parser.parse_args()


def load_vocab(vocab_path):
    """Load vocabulary from file"""
    vocab = text_data.Vocabulary()
    if vocab_path and vocab_path.exists():
        import pickle
        with open(vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            vocab.word2idx = vocab_data['word2idx']
            vocab.idx2word = vocab_data['idx2word']
    return vocab


def predict(model, text, vocab, device, max_seq_len=512):
    """Predict class for a single text"""
    model.eval()
    
    tokens = vocab.tokenize(text)
    indexed = vocab.encode(tokens, max_seq_len)
    input_tensor = torch.tensor(indexed, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, dim=1)
    
    return pred.item()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    
    vocab = load_vocab(args.vocab_path)
    
    model_map = {
        rnn.MODEL_TYPE_LSTM: rnn.LSTMClassifier(
            vocab_size=vocab.size,
            embedding_dim=128,
            hidden_dim=256,
            num_classes=2
        ),
        rnn.MODEL_TYPE_GRU: rnn.GRUClassifier(
            vocab_size=vocab.size,
            embedding_dim=128,
            hidden_dim=256,
            num_classes=2
        ),
        transformer.MODEL_TYPE_TRANSFORMER: transformer.TransformerClassifier(
            vocab_size=vocab.size,
            embedding_dim=128,
            hidden_dim=256,
            num_classes=2
        ),
    }
    
    model = model_map[args.model].to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    if args.text:
        texts = [args.text]
    elif args.text_file and args.text_file.exists():
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        print("Please provide --text or --text-file")
        sys.exit(1)
    
    class_names = ["Negative", "Positive"]
    
    for text in texts:
        text = text.strip()
        if not text:
            continue
        
        pred = predict(model, text, vocab, device, args.max_seq_len)
        print(f"Text: {text[:50]}..." if len(text) > 50 else f"Text: {text}")
        print(f"Prediction: {class_names[pred]} (class {pred})")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)