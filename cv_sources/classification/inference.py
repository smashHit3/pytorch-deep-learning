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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# -----------------------------------------------------------------------------
# Now you can import custom modules using absolute paths from the project root
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
import time

import torch
from PIL import Image
from torchvision import transforms

from cv_sources.models.alexnet import AlexNet

CLASS_NAMES = ["cat", "dog"]


def parse_args():
    parser = ArgumentParser(description="Run AlexNet inference on a single image.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image for inference.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "alexnet.pth",
        help="Path to the saved AlexNet state dictionary.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top predictions to show.",
    )
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = AlexNet(num_classes=len(CLASS_NAMES))
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, top_k: int = 2):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_idx = torch.topk(probabilities, top_k)
    return top_probs.cpu().tolist(), top_idx.cpu().tolist()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print(f"Image path: {args.image}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")

    image_tensor = preprocess_image(args.image, device)
    model = load_model(args.model_path, device)

    start_time = time.time()
    top_probs, top_idx = predict(model, image_tensor, top_k=args.top_k)
    elapsed_ms = (time.time() - start_time) * 1000.0

    print(f"Inference time: {elapsed_ms:.2f} ms")
    print("Top predictions:")
    for prob, idx in zip(top_probs, top_idx):
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
