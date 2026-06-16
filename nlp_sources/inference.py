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
from nlp_sources.models import lstm, gru, transformer


def parse_args():
    parser = ArgumentParser(description="NLP Text Classification Inference")

    parser.add_argument("--model", type=str, required=True,
                        choices=[lstm.MODEL_TYPE_LSTM, gru.MODEL_TYPE_GRU,
                                 transformer.MODEL_TYPE_TRANSFORMER],
                        help="Select model type")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--text", type=str,
                        help="Text to classify")
    parser.add_argument("--text-file", type=Path,
                        help="File containing text to classify")
    parser.add_argument("--dataset", type=str,
                        default=text_data.DATASET_NAME_IMDB,
                        choices=[text_data.DATASET_NAME_IMDB, text_data.DATASET_NAME_AG_NEWS],
                        help="Dataset name (used to rebuild vocabulary)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension (must match training)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (must match training)")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force use CPU")

    return parser.parse_args()


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
    print(f"Using device: {device}")

    # Rebuild vocabulary from the same dataset used for training
    print(f"Rebuilding vocabulary from {args.dataset}...")
    _, _, vocab, num_classes = text_data.load_data(args.dataset, batch_size=32, max_seq_len=args.max_seq_len)
    print(f"Vocabulary size: {vocab.size}")

    # Build model with same architecture as training
    if args.model == lstm.MODEL_TYPE_LSTM:
        model = lstm.lstm_classifier(
            vocab_size=vocab.size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        )
    elif args.model == gru.MODEL_TYPE_GRU:
        model = gru.gru_classifier(
            vocab_size=vocab.size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        )
    elif args.model == transformer.MODEL_TYPE_TRANSFORMER:
        model = transformer.transformer_classifier(
            vocab_size=vocab.size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        )

    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    # Get texts to classify
    if args.text:
        texts = [args.text]
    elif args.text_file and args.text_file.exists():
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        print("Please provide --text or --text-file")
        sys.exit(1)

    # IMDB: Negative/Positive, AG_NEWS: class names
    if args.dataset == text_data.DATASET_NAME_IMDB:
        class_names = ["Negative", "Positive"]
    else:
        class_names = ["World", "Sports", "Business", "Sci/Tech"]

    print(f"\n{'='*60}")
    for text in texts:
        text = text.strip()
        if not text:
            continue

        pred = predict(model, text, vocab, device, args.max_seq_len)
        display_text = text[:60] + "..." if len(text) > 60 else text
        print(f"Text: {display_text}")
        print(f"Prediction: {class_names[pred]} (class {pred})")
        print(f"{'-'*60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
