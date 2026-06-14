#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NLP Training Framework
# -----------------------------------------------------------------------------

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser

from nlp_sources.data_processor import text_data
from nlp_sources.models import lstm, gru, transformer

MODEL_FILE_MAP = {
    lstm.MODEL_TYPE_LSTM: "lstm.pth",
    gru.MODEL_TYPE_GRU: "gru.pth",
    transformer.MODEL_TYPE_TRANSFORMER: "transformer.pth",
}

ORIG_DEFAULT_SAVE_PATH = PROJECT_ROOT / "nlp_sources" / "results" / "default_model.pth"


def parse_args():
    parser = ArgumentParser(description="NLP Text Classification Training Framework")

    # ---------------------- Dataset Config ----------------------
    parser.add_argument("--dataset", type=str, 
                        default=text_data.DATASET_NAME_IMDB,
                        choices=[text_data.DATASET_NAME_IMDB, text_data.DATASET_NAME_AG_NEWS],
                        help="Select training dataset")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloader")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader worker threads")

    # ---------------------- Model Config ----------------------
    parser.add_argument("--model", type=str, default=lstm.MODEL_TYPE_LSTM,
                        choices=[lstm.MODEL_TYPE_LSTM, gru.MODEL_TYPE_GRU, 
                                 transformer.MODEL_TYPE_TRANSFORMER],
                        help="Select NLP model")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for RNN/Transformer")

    # ---------------------- Optimizer Hyperparams ----------------------
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["adam", "sgd"],
                        help="Optimizer type")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="L2 weight decay")

    # ---------------------- Learning Rate Scheduler ----------------------
    parser.add_argument("--use-scheduler", action="store_true", default=True,
                        help="Enable StepLR learning rate scheduler")
    parser.add_argument("--lr-step", type=int, default=None,
                        help="Step interval for LR decay")
    parser.add_argument("--lr-gamma", type=float, default=None,
                        help="LR decay factor")

    # ---------------------- Runtime & Save Config ----------------------
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force use CPU, disable CUDA")
    parser.add_argument("--save-path", type=Path,
                        default=ORIG_DEFAULT_SAVE_PATH,
                        help="Path to save trained model weights")

    return parser.parse_args()


def auto_update_save_path(args):
    if args.save_path == ORIG_DEFAULT_SAVE_PATH:
        model_filename = MODEL_FILE_MAP.get(args.model, "model.pth")
        new_save_path = PROJECT_ROOT / "nlp_sources" / "results" / model_filename
        args.save_path = new_save_path
        print(f"[Auto Update] Default save path changed to: {new_save_path.resolve()}")


def auto_set_model_hyperparams(args):
    model_defaults = {
        lstm.MODEL_TYPE_LSTM: (0.001, 0.0001, "adam", 5, 0.1),
        gru.MODEL_TYPE_GRU: (0.001, 0.0001, "adam", 5, 0.1),
        transformer.MODEL_TYPE_TRANSFORMER: (0.0001, 0.0001, "adam", 5, 0.1),
    }

    default_lr, default_wd, default_opt, default_step, default_gamma = model_defaults.get(
        args.model, (0.001, 0.0, "adam", 5, 0.1)
    )

    if args.lr is None:
        args.lr = default_lr
    if args.weight_decay is None:
        args.weight_decay = default_wd
    if args.optimizer is None:
        args.optimizer = default_opt
    if args.lr_step is None:
        args.lr_step = default_step
    if args.lr_gamma is None:
        args.lr_gamma = default_gamma


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_type: str, vocab_size: int, num_classes: int, args):
    model_map = {
        lstm.MODEL_TYPE_LSTM: lambda: lstm.lstm_classifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        ),
        gru.MODEL_TYPE_GRU: lambda: gru.gru_classifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        ),
        transformer.MODEL_TYPE_TRANSFORMER: lambda: transformer.transformer_classifier(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes
        ),
    }
    return model_map[model_type]()


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        _, preds = torch.max(outputs, dim=1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += texts.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * texts.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += texts.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_loop(model, device, train_loader, val_loader, optimizer, scheduler, criterion, epochs):
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, device, val_loader, criterion)

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if scheduler is not None:
            scheduler.step()


def save_weights(model: nn.Module, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model weights saved to: {save_path.resolve()}")


def main():
    args = parse_args()

    auto_set_model_hyperparams(args)
    auto_update_save_path(args)

    print(f"[Auto Config] Model: {args.model} | LR: {args.lr} | Weight Decay: {args.weight_decay}")

    set_random_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}\n")

    train_loader, val_loader, vocab, num_classes = text_data.load_data(
        args.dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len
    )
    print(f"Vocabulary size: {vocab.size}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}\n")

    model = build_model(args.model, vocab.size, num_classes, args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) if args.use_scheduler else None

    train_loop(model, device, train_loader, val_loader, optimizer, scheduler, criterion, args.epochs)

    save_weights(model, args.save_path)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("\n❌ ERROR: Dataset not found. Please check the dataset path.")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("\n❌ ERROR: CUDA Out of Memory! Reduce --batch-size or --max-seq-len.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        sys.exit(1)