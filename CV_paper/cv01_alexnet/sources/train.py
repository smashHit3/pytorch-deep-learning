from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tools.data_processor import get_data_loaders
from alexnet import AlexNet


def parse_args():
    parser = ArgumentParser(description="Train AlexNet on the cats-vs-dogs dataset.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "dataset",
                        help="Root directory containing dogs_vs_cats or dataset/dogs_vs_cats.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay for optimizer.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory for DataLoader.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--save-path", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "alexnet.pth",
                        help="Path to save the trained model state.")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available.")
    return parser.parse_args()


def train_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    train_loader, val_loader, _ = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=args.seed,
        verbose=True,
    )

    model = AlexNet(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        scheduler.step()

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(args.save_path))
    print(f"Saved AlexNet weights to {args.save_path}")


if __name__ == "__main__":
    main()
