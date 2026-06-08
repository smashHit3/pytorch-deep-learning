from pathlib import Path
from data_processor.fashion_mnist import load_data_fashion_mnist
from data_processor.dogs_vs_cats import load_data_dogs_vs_cats
from argparse import ArgumentParser
from models.googlenet import GoogleNet
import torch


def parse_args():
    parser = ArgumentParser(description="Train a specified model on the specified dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin memory for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-path", type=Path, default=Path(__file__).resolve().parent / "results" / "alexnet.pth",
                        help="Path to save the trained model state.")
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


def train_googlenet(model, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


def load_dataset(dataset_path, batch_size):
    train_loader, val_loader = load_data_fashion_mnist(batch_size, resize=224)
    return train_loader, val_loader, None


def load_model():
    model = GoogleNet(num_classes=10, init_weights=True)
    return model


def save_model(model, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved GoogleNet weights to {save_path}")


def main():
    args = parse_args()
    train_loader, val_loader, _ = load_dataset("./", args.batch_size)
    model = load_model()
    train_googlenet(model, train_loader, val_loader)
    save_model(model, args.save_path)


if __name__ == "__main__":
    main()