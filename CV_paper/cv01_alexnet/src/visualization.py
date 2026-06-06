from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image

from alexnet import AlexNet


def parse_args():
    parser = ArgumentParser(description="Visualize AlexNet kernels and feature maps.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image for feature map visualization.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "alexnet.pth",
        help="Path to the saved AlexNet state dictionary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "alexnet_visualization",
        help="Directory to write TensorBoard visualization logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to use for visualization.",
    )
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = AlexNet(num_classes=2)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def visualize_kernels(model: nn.Module, writer: SummaryWriter):
    kernel_index = 0
    for module in model.modules():
        if not isinstance(module, nn.Conv2d):
            continue

        kernels = module.weight.detach().cpu()
        print(f"conv{kernel_index} kernels shape: {kernels.shape}")

        channels_out, channels_in, kernel_h, kernel_w = kernels.shape
        for i in range(channels_out):
            kernel = kernels[i].unsqueeze(1)
            kernel_grid = vutils.make_grid(kernel, nrow=8, normalize=True, scale_each=True)
            writer.add_image(f"conv{kernel_index}_kernel_{i}", kernel_grid, global_step=i)

        kernels_all = kernels.view(-1, 1, kernel_h, kernel_w)
        kernels_grid = vutils.make_grid(kernels_all, nrow=8, normalize=True, scale_each=True)
        writer.add_image(f"conv{kernel_index}_kernels_all", kernels_grid, global_step=0)
        kernel_index += 1


def visualize_feature_map(model: nn.Module, image_path: Path, device: torch.device, writer: SummaryWriter):
    transform = build_transform()
    image_rgb = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        raise RuntimeError("No Conv2d layer found in model for feature map visualization.")

    with torch.no_grad():
        feature_map = first_conv(image_tensor).cpu()

    feature_grid = vutils.make_grid(feature_map.transpose(1, 0), nrow=8, normalize=True, scale_each=True)
    writer.add_image("conv1_feature_map", feature_grid, global_step=0)
    print(f"conv1 feature map shape: {feature_map.shape}")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"image: {args.image}")
    print(f"model path: {args.model_path}")
    print(f"output dir: {args.output_dir}")
    print(f"device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.output_dir), filename_suffix="_alexnet")

    model = load_model(args.model_path, device)
    visualize_kernels(model, writer)
    visualize_feature_map(model, args.image, device, writer)

    writer.close()
    print("Visualization logs written to:", args.output_dir)


if __name__ == "__main__":
    main()
