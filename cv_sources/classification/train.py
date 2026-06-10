# -----------------------------------------------------------------------------
# Purpose: Add the project root directory to Python's system path (sys.path)
# 
# This solves the "ModuleNotFoundError" when importing custom modules from other 
# directories in your project (e.g., data_processor/fashion_mnist.py, models/googlenet.py).
# 
# Python only searches for modules in directories listed in `sys.path` by default.
# By adding the project root to `sys.path`, we enable absolute imports from any script
# in the project, regardless of where the script is executed from.
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))
# -----------------------------------------------------------------------------
# Now you can import custom modules using absolute paths from the project root
# -----------------------------------------------------------------------------

from cv_sources.data_processor import fashion_mnist
from cv_sources.data_processor import dogs_vs_cats
from argparse import ArgumentParser
from cv_sources.models import alexnet
from cv_sources.models import googlenet
from cv_sources.models import vgg
import torch


def parse_args():
    parser = ArgumentParser(description="Train a specified model on the specified dataset")
    parser.add_argument("--dataset-name", type=str, default="dogs_vs_cats", 
                        choices=[dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS, fashion_mnist.DATASET_NAME_FASHION_MNIST],
                        help="Data name for selecting to train which dataset")
    parser.add_argument("--model-type", type=str, default=alexnet.MODEL_TYPE_ALEXNET, 
                        choices=[alexnet.MODEL_TYPE_ALEXNET, vgg.MODEL_TYPE_VGG11, vgg.MODEL_TYPE_VGG13, 
                                 vgg.MODEL_TYPE_VGG16, vgg.MODEL_TYPE_VGG19, googlenet.MODEL_TYPE_GOOGLENET], 
                        help="Model type for selecting to train which model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin memory for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-path", type=Path, default=PROJECT_ROOT / "results" / "alexnet.pth",
                        help="Path to save the trained model state.")
    return parser.parse_args()


def train_epoch(model: torch.nn.Module, device: torch.device, 
                train_loader: torch.utils.data.DataLoader, criterion, optimizer):
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


def validate(model: torch.nn.Module, device: torch.device, 
             val_loader: torch.utils.data.DataLoader, criterion):
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


def train_model(model: torch.nn.Module, device: torch.device, 
                    train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


def load_dataset(dataset_name: Path, batch_size: int, resize: int=224):
    if dataset_name == fashion_mnist.DATASET_NAME_FASHION_MNIST:
        train_loader, val_loader = fashion_mnist.load_data_fashion_mnist(batch_size, resize=resize)
        return train_loader, val_loader, None, 10
    
    train_loader, val_loader, test_loader = dogs_vs_cats.load_data_dogs_vs_cats(batch_size=batch_size, resize=resize)
    return train_loader, val_loader, test_loader, 2


def load_model(model_type: str, num_classes: int):
    if model_type == alexnet.MODEL_TYPE_ALEXNET:
        model = alexnet.AlexNet(num_classes=num_classes)
    elif model_type == vgg.MODEL_TYPE_VGG11:
        model = vgg.vgg11(num_classes=num_classes, init_weights=True)
    elif model_type == vgg.MODEL_TYPE_VGG13:
        model = vgg.vgg13(num_classes=num_classes, init_weights=True)
    elif model_type == vgg.MODEL_TYPE_VGG16:
        model = vgg.vgg16(num_classes=num_classes, init_weights=True)
    elif model_type == vgg.MODEL_TYPE_VGG19:
        model = vgg.vgg19(num_classes=num_classes, init_weights=True)
    elif model_type == googlenet.MODEL_TYPE_GOOGLENET:
        model = googlenet.GoogleNet(num_classes=num_classes, init_weights=True)
    return model


def save_model(model: torch.nn.Module, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved GoogleNet weights to {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    train_loader, val_loader, _, num_classes = load_dataset(args.dataset_name, args.batch_size)
    model = load_model(args.model_type, num_classes)
    train_model(model, device, train_loader, val_loader)
    save_model(model, args.save_path)


if __name__ == "__main__":
    main()