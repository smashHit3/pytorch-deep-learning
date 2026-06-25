# -----------------------------------------------------------------------------
# Add project root to system path
# -----------------------------------------------------------------------------
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser

# Import custom modules
from cv_sources.data_processor import fashion_mnist, dogs_vs_cats
from cv_sources.classification.train import build_model, MODEL_FILE_MAP

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

# Labels for datasets
LABELS = {
    fashion_mnist.DATASET_NAME_FASHION_MNIST: [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ],
    dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS: ["Cat", "Dog"]
}

DATASET_NUM_CLASSES = {
    fashion_mnist.DATASET_NAME_FASHION_MNIST: len(LABELS[fashion_mnist.DATASET_NAME_FASHION_MNIST]),
    dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS: len(LABELS[dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS]),
}

MODEL_OUTPUT_WEIGHT_KEYS = {
    "alexnet": "classifier.6.weight",
    "googlenet": "fc.weight",
    "vgg11": "classifier.6.weight",
    "vgg13": "classifier.6.weight",
    "vgg16": "classifier.6.weight",
    "vgg19": "classifier.6.weight",
    "resnet18": "fc.weight",
    "resnet34": "fc.weight",
    "resnet50": "fc.weight",
    "densenet121": "fc.weight",
    "densenet169": "fc.weight",
    "densenet201": "fc.weight",
    "mobilenet_1_0": "classifier.1.weight",
    "mobilenet_0_5": "classifier.1.weight",
    "mobilenet_0_75": "classifier.1.weight",
}

# 1. Record original default weight path (for judgment)
ORIG_DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "results" / "default_model.pth"


def auto_update_model_path(args) -> None:
    """
    Auto update default weight path by selected model
    Rule: Only modify when user uses the original default path
    Manual --model-path will NOT be overwritten
    """
    if args.model_path == ORIG_DEFAULT_WEIGHT_PATH:
        # Get matched weight filename from MODEL_FILE_MAP in train.py
        weight_file = MODEL_FILE_MAP.get(args.model, "model.pth")
        new_weight_path = PROJECT_ROOT / "results" / weight_file
        args.model_path = new_weight_path
        print(f"[Auto Update] Default weight path changed to: {new_weight_path.resolve()}")


def load_model_metadata(weight_path: Path) -> Dict[str, Any]:
    """Load optional model metadata saved alongside a checkpoint."""
    metadata_path = weight_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(weight_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint or state_dict."""
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def infer_num_classes_from_state_dict(model_name: str, state_dict: Dict[str, Any]) -> int:
    """Infer the output class count from the classifier layer in a state dict."""
    weight_key = MODEL_OUTPUT_WEIGHT_KEYS.get(model_name)
    if weight_key is None or weight_key not in state_dict:
        raise KeyError(f"Unable to infer class count for model '{model_name}' from checkpoint.")
    return int(state_dict[weight_key].shape[0])


def resolve_inference_settings(args) -> Dict[str, Any]:
    """Resolve dataset, class count, labels, and checkpoint metadata for inference."""
    metadata = load_model_metadata(args.model_path)
    checkpoint_dataset = metadata.get("dataset")

    if args.dataset and checkpoint_dataset and args.dataset != checkpoint_dataset:
        raise ValueError(
            f"Checkpoint dataset mismatch: requested '{args.dataset}' but weights were trained for '{checkpoint_dataset}'."
        )

    resolved_dataset = args.dataset or checkpoint_dataset
    state_dict = load_checkpoint(args.model_path, torch.device("cpu"))
    inferred_num_classes = infer_num_classes_from_state_dict(args.model, state_dict)

    if resolved_dataset in DATASET_NUM_CLASSES and DATASET_NUM_CLASSES[resolved_dataset] != inferred_num_classes:
        raise ValueError(
            f"Dataset '{resolved_dataset}' expects {DATASET_NUM_CLASSES[resolved_dataset]} classes, "
            f"but checkpoint outputs {inferred_num_classes}."
        )

    if metadata.get("num_classes") is not None and int(metadata["num_classes"]) != inferred_num_classes:
        raise ValueError(
            f"Checkpoint metadata mismatch: metadata num_classes={metadata['num_classes']} "
            f"but weights output {inferred_num_classes} classes."
        )

    if resolved_dataset in DATASET_NUM_CLASSES:
        num_classes = DATASET_NUM_CLASSES[resolved_dataset]
        class_names = LABELS[resolved_dataset]
    else:
        num_classes = inferred_num_classes
        class_names = args.class_names or [f"Class_{idx}" for idx in range(num_classes)]

    if args.class_names and len(args.class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(args.class_names)}.")

    if args.class_names:
        class_names = args.class_names

    return {
        "dataset": resolved_dataset,
        "num_classes": num_classes,
        "class_names": class_names,
        "state_dict": state_dict,
        "metadata": metadata,
    }


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
    if args.resize <= 0 or args.crop <= 0:
        raise ValueError("Image resize/crop size must be positive integer")


def parse_args() -> object:
    """Parse command line arguments"""
    parser = ArgumentParser(description="Universal CV Model Inference Pipeline")

    # 1. Input Image Config
    parser.add_argument("--image", type=Path, required=True, help="Path to input image file")

    # 2. Dataset Config (to auto-configure preprocessing)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=[fashion_mnist.DATASET_NAME_FASHION_MNIST,
                                 dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS],
                        help="Select dataset for automatic preprocessing config")

    # 3. Model Config
    parser.add_argument("--model", type=str, default="alexnet", choices=list(MODEL_FILE_MAP.keys()),
                        help="Select model (must match MODEL_TYPE in train.py)")
    parser.add_argument("--model-path", type=Path,
                        default=ORIG_DEFAULT_WEIGHT_PATH,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--num-classes", type=int, default=None, help="Total number of classification categories")

    # 4. Image Preprocessing Config
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help="Image resize size before center crop")
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP, help="Final center crop size for model input")
    parser.add_argument("--mean", nargs=3, type=float, default=DEFAULT_MEAN,
                        help="Normalization mean (R G B)")
    parser.add_argument("--std", nargs=3, type=float, default=DEFAULT_STD,
                        help="Normalization std (R G B)")

    # 5. Inference Config
    parser.add_argument("--top-k", type=int, default=2, help="Output top K prediction results")
    parser.add_argument("--class-names", nargs="+", default=None,
                        help="List of category names, e.g. --class-names cat dog")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")

    # 6. Device & Performance Config
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(
    img_path: Path,
    dataset: str = None,
    resize_size: int = DEFAULT_RESIZE,
    crop_size: int = DEFAULT_CROP,
    mean: Tuple[float, float, float] = DEFAULT_MEAN,
    std: Tuple[float, float, float] = DEFAULT_STD,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Load and preprocess image based on dataset or manual params"""
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")

    if dataset == fashion_mnist.DATASET_NAME_FASHION_MNIST:
        # Match fashion_mnist.py: Resize -> Grayscale(3) -> ToTensor
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    elif dataset == dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS:
        # Match dogs_vs_cats.py val_tf: Resize(256) -> CenterCrop(crop_size) -> ToTensor -> Normalize
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Manual fallback
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    img_tensor = transform(img_rgb).unsqueeze(0)
    return img_tensor.to(device)


def load_model(
    model_name: str,
    weight_path: Optional[Path],
    num_classes: int,
    device: torch.device,
    state_dict: Optional[Dict[str, Any]] = None,
    use_fp16: bool = False
) -> torch.nn.Module:
    """Load model architecture and weights, validating the checkpoint class count."""
    if state_dict is None:
        if weight_path is None:
            raise ValueError("weight_path is required when state_dict is not provided.")
        state_dict = load_checkpoint(weight_path, device)

    inferred_num_classes = infer_num_classes_from_state_dict(model_name, state_dict)
    if inferred_num_classes != num_classes:
        raise ValueError(
            f"Checkpoint for model '{model_name}' outputs {inferred_num_classes} classes, not {num_classes}."
        )

    # Use build_model from train.py to ensure consistency
    model = build_model(model_name, num_classes, init_weights=False)
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
        # Match dtype of model parameters
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
        resolved = resolve_inference_settings(args)
        if args.num_classes is not None and args.num_classes != resolved["num_classes"]:
            raise ValueError(
                f"Requested --num-classes {args.num_classes}, but checkpoint requires {resolved['num_classes']}."
            )
        if args.top_k <= 0 or args.top_k > resolved["num_classes"]:
            raise ValueError(f"top-k must be between 1 and {resolved['num_classes']}")
        # Set random seed
        set_random_seed(args.seed)
        # Select device
        device = get_device(args.device)

        print(f"==================== Inference Config ====================")
        print(f"Image Path: {args.image.resolve()}")
        print(f"Dataset: {resolved['dataset'] if resolved['dataset'] else 'Manual'}")
        print(f"Model: {args.model} | Weights: {args.model_path.resolve()}")
        print(f"Device: {device.type.upper()}")
        print(f"Top-K: {args.top_k} | Classes: {resolved['num_classes']}")
        print("==========================================================\n")

        # Preprocess image
        img_tensor = preprocess_image(
            img_path=args.image,
            dataset=resolved["dataset"],
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
            num_classes=resolved["num_classes"],
            device=device,
            state_dict=resolved["state_dict"],
            use_fp16=args.fp16
        )

        # Run inference
        start_time = time.perf_counter()
        top_probs, top_indices = run_inference(model, img_tensor, args.top_k)
        infer_latency_ms = (time.perf_counter() - start_time) * 1000

        # Determine class names
        final_class_names = resolved["class_names"]

        # Print results
        print(f"⏱️  Inference Latency: {infer_latency_ms:.2f} ms")
        print("\n🏆 Top Prediction Results:")
        for prob, idx in zip(top_probs, top_indices):
            class_name = final_class_names[idx] if idx < len(final_class_names) else f"Class_{idx}"
            print(f"  {class_name:<6} | Confidence: {prob:.4f}")

    except Exception as e:
        print(f"\n❌ Runtime Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
