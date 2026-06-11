# -----------------------------------------------------------------------------
# Add project root to system path
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser

# Import custom modules
from cv_sources.data_processor import fashion_mnist, dogs_vs_cats
from cv_sources.models import alexnet, googlenet, vgg

# ---------------- Global Constant: Model -> Save Filename Mapping ----------------
# Match model type to weight file name for auto path update
MODEL_FILE_MAP = {
    alexnet.MODEL_TYPE_ALEXNET: "alexnet.pth",
    vgg.MODEL_TYPE_VGG11: "vgg11.pth",
    vgg.MODEL_TYPE_VGG13: "vgg13.pth",
    vgg.MODEL_TYPE_VGG16: "vgg16.pth",
    vgg.MODEL_TYPE_VGG19: "vgg19.pth",
    googlenet.MODEL_TYPE_GOOGLENET: "googlenet.pth"
}
# Original default save path (for judgment)
ORIG_DEFAULT_SAVE_PATH = PROJECT_ROOT / "results" / "model.pth"


def parse_args():
    parser = ArgumentParser(description="Unified CV Training Framework (AlexNet/VGG/GoogLeNet)")

    # ---------------------- Dataset Config ----------------------
    parser.add_argument("--dataset", type=str, required=False,
                        default=dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS,
                        choices=[fashion_mnist.DATASET_NAME_FASHION_MNIST,
                                 dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS],
                        help="Select training dataset")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Input image resize size")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for dataloader")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader worker threads")
    parser.add_argument("--pin-memory", action="store_true", default=True,
                        help="Enable pin memory for data loading")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                        help="Disable pin memory")

    # ---------------------- Model Config ----------------------
    parser.add_argument("--model", type=str, default=alexnet.MODEL_TYPE_ALEXNET,
                        choices=[alexnet.MODEL_TYPE_ALEXNET, googlenet.MODEL_TYPE_GOOGLENET,
                                 vgg.MODEL_TYPE_VGG11, vgg.MODEL_TYPE_VGG13,
                                 vgg.MODEL_TYPE_VGG16, vgg.MODEL_TYPE_VGG19],
                        help="Select CNN model (AlexNet as default)")
    parser.add_argument("--init-weights", action="store_true", default=True,
                        help="Initialize model weights")
    parser.add_argument("--no-init-weights", dest="init-weights", action="store_false",
                        help="Disable weight initialization")

    # ---------------------- Optimizer & AlexNet Hyperparams ----------------------
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate (Auto set for AlexNet: 0.01)")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["adam", "sgd"],
                        help="Optimizer type (AlexNet official: SGD)")
    parser.add_argument("--momentum", type=float, default=None,
                        help="Momentum for SGD (Auto set for AlexNet: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="L2 weight decay (Auto set for AlexNet: 5e-4)")

    # ---------------------- Learning Rate Scheduler ----------------------
    parser.add_argument("--use-scheduler", action="store_true", default=True,
                        help="Enable StepLR learning rate scheduler")
    parser.add_argument("--lr-step", type=int, default=5,
                        help="Step interval for LR decay")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="LR decay factor (multiply lr by gamma every lr-step epochs)")

    # ---------------------- Runtime & Save Config ----------------------
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force use CPU, disable CUDA")
    parser.add_argument("--save-path", type=Path,
                        default=ORIG_DEFAULT_SAVE_PATH,
                        help="Path to save trained model weights (auto rename if use default)")

    return parser.parse_args()


def auto_update_save_path(args):
    """
    Auto update default save path based on selected model
    Rule: Only modify when user uses the original default path
          Manual --save-path will NOT be overwritten
    """
    # Judge: whether user is using the original default save path
    if args.save_path == ORIG_DEFAULT_SAVE_PATH:
        # Get corresponding filename by model type
        model_filename = MODEL_FILE_MAP.get(args.model, "model.pth")
        # Rebuild new save path (keep results/ directory, change filename)
        new_save_path = PROJECT_ROOT / "results" / model_filename
        args.save_path = new_save_path
        print(f"[Auto Update] Default save path changed to: {new_save_path.resolve()}")


def auto_set_alexnet_hyperparams(args):
    """
    Auto apply AlexNet official paper hyperparameters
    If user does NOT manually set params, overwrite with AlexNet standard values
    """
    if args.model == alexnet.MODEL_TYPE_ALEXNET:
        if args.lr is None:
            args.lr = 0.01
        if args.momentum is None:
            args.momentum = 0.9
        if args.weight_decay is None:
            args.weight_decay = 5e-4
    # Set default for other models if missing
    else:
        if args.lr is None:
            args.lr = 0.001
        if args.momentum is None:
            args.momentum = 0.9
        if args.weight_decay is None:
            args.weight_decay = 0.0


def set_random_seed(seed: int):
    """Set global random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(args):
    """Load dataset and dataloaders by parameter"""
    if args.dataset == fashion_mnist.DATASET_NAME_FASHION_MNIST:
        train_loader, val_loader = fashion_mnist.load_data_fashion_mnist(
            batch_size=args.batch_size,
            resize=args.img_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        num_classes = 10
    else:
        train_loader, val_loader, _ = dogs_vs_cats.load_data_dogs_vs_cats(
            batch_size=args.batch_size,
            resize=args.img_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        num_classes = 2

    return train_loader, val_loader, num_classes


def build_model(model_type: str, num_classes: int, init_weights: bool):
    """Build model by parameter"""
    model_map = {
        alexnet.MODEL_TYPE_ALEXNET: lambda: alexnet.AlexNet(num_classes=num_classes, init_weights=init_weights),
        googlenet.MODEL_TYPE_GOOGLENET: lambda: googlenet.GoogleNet(num_classes=num_classes, init_weights=init_weights),
        vgg.MODEL_TYPE_VGG11: lambda: vgg.vgg11(num_classes=num_classes, init_weights=init_weights),
        vgg.MODEL_TYPE_VGG13: lambda: vgg.vgg13(num_classes=num_classes, init_weights=init_weights),
        vgg.MODEL_TYPE_VGG16: lambda: vgg.vgg16(num_classes=num_classes, init_weights=init_weights),
        vgg.MODEL_TYPE_VGG19: lambda: vgg.vgg19(num_classes=num_classes, init_weights=init_weights)
    }
    return model_map[model_type]()


def build_optimizer(model: nn.Module, opt_type: str, lr: float, momentum: float, weight_decay: float):
    """Build optimizer with L2 weight decay (compatible with AlexNet)"""
    if opt_type.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    return optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


def build_scheduler(optimizer, use_scheduler: bool, lr_step: int, lr_gamma: float):
    """Create StepLR scheduler"""
    if use_scheduler:
        return StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    return None


def train_one_epoch(model, device, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_loop(model, device, train_loader, val_loader, optimizer, scheduler, criterion, epochs):
    """Main training loop + LR scheduler update each epoch"""
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, device, val_loader, criterion)

        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Update learning rate scheduler every epoch
        if scheduler is not None:
            scheduler.step()


def save_weights(model: nn.Module, save_path: Path):
    """Save model state dict"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model weights saved to: {save_path.resolve()}")


def main():
    args = parse_args()

    # 1. Auto fill AlexNet official hyperparameters
    auto_set_alexnet_hyperparams(args)
    # 2. Auto update default save path according to selected model (CORE NEW LOGIC)
    auto_update_save_path(args)

    print(f"[Auto Config] Model: {args.model} | LR: {args.lr} | Momentum: {args.momentum} | Weight Decay: {args.weight_decay}")

    # Set random seed
    set_random_seed(args.seed)

    # Device setup
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}\n")

    # Load dataset
    train_loader, val_loader, num_classes = load_dataset(args)

    # Build model
    model = build_model(args.model, num_classes, args.init_weights).to(device)

    # Build loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        args.optimizer,
        args.lr,
        args.momentum,
        args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.use_scheduler, args.lr_step, args.lr_gamma)

    # Start training
    train_loop(model, device, train_loader, val_loader, optimizer, scheduler, criterion, args.epochs)

    # Save model
    save_weights(model, args.save_path)


if __name__ == "__main__":
    # Global exception handling
    try:
        main()
    except FileNotFoundError:
        print("\n❌ ERROR: Dataset or save directory not found. Check file paths.")
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("\n❌ ERROR: CUDA Out of Memory! Reduce --batch-size.")
        sys.exit(1)
    except PermissionError:
        print("\n❌ ERROR: Permission denied. Cannot save model file.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        sys.exit(1)