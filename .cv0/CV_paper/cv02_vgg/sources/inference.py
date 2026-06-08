from argparse import ArgumentParser
from pathlib import Path
import torch
from vgg import vgg16
from torchvision import transforms
from PIL import Image
import time

# we have two classes now cat:0 and dog:1
CLASS_NAMES = ["cat", "dag"]

# need implement parse arguments, such as --image, --model-path, --device, --top-k
def parse_args():
    parser = ArgumentParser(description="Run VGG inference on a single image.")
    parser.add_argument("--image", type=Path, required=True, 
                        help="Path to the input image for inference.")
    parser.add_argument("--model-path", type=Path, 
                        default=Path(__file__).resolve().parents[1] / "results" / "vgg16.pth", 
                        help="Path to the saved vgg state dictionary.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        choices=["cuda", "cpu"], help="Device to use for inference.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of top predictions to show.")
    return parser.parse_args()


# implement loading model
def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    # check the model path whether is valid
    if not model_path.exists():
        return FileNotFoundError(f"Model file not found: {model_path}")
    # new a model
    model = vgg16(num_classes=len(CLASS_NAMES))
    state = torch.load(model_path, map_location=device)
    # check "state_dict" in state
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # load the weights
    model.load_state_dict(state_dict=state)
    # move to device
    model.to(device)
    # change to evaluation
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
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    
    print(f"Image path: {args.image}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {device}")

    image_tensor = preprocess_image(args.image, device)
    model = load_model(args.model_path, device)

    start_time = time.time()
    top_probs, top_idx = predict(model, image_tensor, args.top_k)
    elapsed_ms = (time.time() - start_time) * 1000.0

    print(f"Inference time: {elapsed_ms:.2f} ms")
    print("Top predictions:")
    for prob, idx in zip(top_probs, top_idx):
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()