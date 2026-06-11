import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser

# -------------------------- Project Path Configuration --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

# Import custom models
from cv_sources.models.alexnet import AlexNet
from cv_sources.models.vgg import vgg11, vgg13, vgg16, vgg19
from cv_sources.models.googlenet import GoogleNet

# -------------------------- Global Config (Aligned with Training Script) --------------------------
# Default ImageNet normalization params
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
# Default preprocessing size
DEFAULT_RESIZE = 256
DEFAULT_CROP = 224
# Default class names (cats vs dogs)
DEFAULT_CLASS_NAMES = ["cat", "dog"]
# Default random seed
DEFAULT_SEED = 42

# 1. Model -> Model Class/Function mapping
MODEL_FACTORY = {
    "alexnet": AlexNet,
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "googlenet": GoogleNet
}

# 2. Model -> Default Weight File Name (SAME as training script)
MODEL_WEIGHT_MAP = {
    "alexnet": "alexnet.pth",
    "vgg11": "vgg11.pth",
    "vgg13": "vgg13.pth",
    "vgg16": "vgg16.pth",
    "vgg19": "vgg19.pth",
    "googlenet": "googlenet.pth"
}

# 3. Record original default weight path (for judgment)
ORIG_DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "results" / "default_model.pth"


def auto_update_model_path(args) -> None:
    """
    Auto update default weight path by selected model
    Rule: Only modify when user uses the original default path
          Manual --model-path will NOT be overwritten
    """
    if args.model_path == ORIG_DEFAULT_WEIGHT_PATH:
        # Get matched weight filename
        weight_file = MODEL_WEIGHT_MAP.get(args.model, "model.pth")
        new_weight_path = PROJECT_ROOT / "results" / weight_file
        args.model_path = new_weight_path
        print(f"[Auto Update] Default weight path changed to: {new_weight_path.resolve()}")


def set_random_seed(seed: int) -> None:
    """Fix random seed for reproducible inference results"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_args(args) -> None:
    """Validate input parameters to avoid runtime errors"""
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model weight file not found: {args.model_path}")
    if args.top_k <= 0 or args.top_k > args.num_classes:
        raise ValueError(f"top-k must be between 1 and {args.num_classes}")
    if args.resize <= 0 or args.crop <= 0:
        raise ValueError("Image resize/crop size must be positive integer")


def parse_args() -> object:
    """Parse command line arguments"""
    parser = ArgumentParser(description="Universal CV Model Inference Pipeline")

    # 1. Input Image Config
    parser.add_argument("--image", type=Path, required=True, help="Path to input image file")

    # 2. Model Config
    parser.add_argument("--model", type=str, default="alexnet", choices=list(MODEL_FACTORY.keys()),
                        help="Select model: alexnet / vgg11 / vgg13 / vgg16 / vgg19 / googlenet")
    parser.add_argument("--model-path", type=Path,
                        default=ORIG_DEFAULT_WEIGHT_PATH,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--num-classes", type=int, default=2, help="Total number of classification categories")

    # 3. Image Preprocessing Config
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help="Image resize size before center crop")
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP, help="Final center crop size for model input")
    parser.add_argument("--mean", nargs=3, type=float, default=DEFAULT_MEAN,
                        help="Normalization mean (R G B)")
    parser.add_argument("--std", nargs=3, type=float, default=DEFAULT_STD,
                        help="Normalization std (R G B)")

    # 4. Inference Config
    parser.add_argument("--top-k", type=int, default=2, help="Output top K prediction results")
    parser.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES,
                        help="List of category names, e.g. --class-names cat dog")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")

    # 5. Device & Performance Config
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Running device: auto(auto select), cpu, cuda")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 half-precision (CUDA only, speed up inference)")

    return parser.parse_args()


def get_device(device_opt: str) -> torch.device:
    """Unified device selection logic"""
    if device_opt == "cpu":
        return torch.device("cpu")
    elif device_opt == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, fallback to CPU")
            return torch.device("cpu")
    # Auto mode
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(
    img_path: Path,
    resize_size: int,
    crop_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    device: torch.device
) -> torch.Tensor:
    """Load and preprocess single image to model input tensor"""
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
    img_tensor = transform(img_rgb).unsqueeze(0)
    return img_tensor.to(device)


def load_model(
    model_name: str,
    weight_path: Path,
    num_classes: int,
    device: torch.device,
    use_fp16: bool = False
) -> torch.nn.Module:
    """Load model and weights, support FP16 half-precision"""
    model_cls = MODEL_FACTORY[model_name]
    model = model_cls(num_classes=num_classes)

    # Load state dict (support both raw state_dict and checkpoint wrapper)
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    if use_fp16 and device.type == "cuda":
        model.half()
        print("✅ FP16 half-precision enabled for acceleration")

    return model


def run_inference(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    top_k: int
) -> Tuple[List[float], List[int]]:
    """Run forward inference and get top-K results"""
    with torch.no_grad():
        # 🔧 FIX: Get dtype from model parameters instead of model object
        if next(model.parameters()).dtype == torch.float16:
            img_tensor = img_tensor.half()

        output = model(img_tensor)
        prob = F.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(prob, k=top_k)

    return top_probs.cpu().tolist(), top_indices.cpu().tolist()


def main():
    try:
        args = parse_args()

        # Core fix: Auto switch weight path by model name
        auto_update_model_path(args)

        # Validate all arguments
        validate_args(args)
        # Set random seed
        set_random_seed(args.seed)
        # Select device
        device = get_device(args.device)

        print(f"==================== Inference Config ====================")
        print(f"Image Path: {args.image.resolve()}")
        print(f"Model: {args.model} | Weights: {args.model_path.resolve()}")
        print(f"Device: {device.type.upper()}")
        print(f"Top-K: {args.top_k} | Classes: {args.num_classes}")
        print("==========================================================\n")

        # Preprocess image
        img_tensor = preprocess_image(
            img_path=args.image,
            resize_size=args.resize,
            crop_size=args.crop,
            mean=args.mean,
            std=args.std,
            device=device
        )

        # Load model
        model = load_model(
            model_name=args.model,
            weight_path=args.model_path,
            num_classes=args.num_classes,
            device=device,
            use_fp16=args.fp16
        )

        # Run inference
        start_time = time.perf_counter()
        top_probs, top_indices = run_inference(model, img_tensor, args.top_k)
        infer_latency_ms = (time.perf_counter() - start_time) * 1000

        # Print results
        print(f"⏱️  Inference Latency: {infer_latency_ms:.2f} ms")
        print("\n🏆 Top Prediction Results:")
        for prob, idx in zip(top_probs, top_indices):
            class_name = args.class_names[idx] if idx < len(args.class_names) else f"Class_{idx}"
            print(f"  {class_name:<6} | Confidence: {prob:.4f}")

    except Exception as e:
        print(f"\n❌ Runtime Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()