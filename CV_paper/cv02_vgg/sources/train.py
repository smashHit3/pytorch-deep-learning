from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn, optim
from vgg import vgg11
from vgg import vgg13
from vgg import vgg16
from vgg import vgg19
from tools.data_processor import get_data_loaders

def parse_args():
    parser = ArgumentParser(description="Train a VGG model on the Dogs vs Cats dataset")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "dataset", help="Root directory of the dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin memory for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-path", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "vgg16.pth",
                        help="Path to save the trained model state.")
    parser.add_argument("--vgg-version", type=str, default="vgg16", help="Version of VGG model to train")
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
    print(f"Using device: {device}")
    train_loader, val_loader, _ = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=args.seed,
    )
    vgg_version = args.vgg_version
    if vgg_version == "vgg11":
        model = vgg11(num_classes=2, init_weights=True).to(device)
    elif vgg_version == "vgg13":
        model = vgg13(num_classes=2, init_weights=True).to(device)
    elif vgg_version == "vgg16":
        model = vgg16(num_classes=2, init_weights=True).to(device)
    elif vgg_version == "vgg19":
        model = vgg19(num_classes=2, init_weights=True).to(device)
    else:
        raise ValueError("Invalid VGG version")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(args.save_path))
    print(f"Saved VGG weights to {args.save_path}")

if __name__ == "__main__":
    main()