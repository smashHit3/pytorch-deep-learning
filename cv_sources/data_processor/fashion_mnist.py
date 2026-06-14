"""
FashionMNIST dataset
@File: fashion_mnist.py
@Description: Data loading for FashionMNIST with auto-download
"""

import torchvision
from torchvision import transforms
from torch.utils import data
from cv_sources.data_processor.base import _project_root

DATASET_NAME_FASHION_MNIST = "fashion_mnist"
NUM_CLASSES = 10


def load_data_fashion_mnist(batch_size: int, resize: int = None, num_workers: int = 4, pin_memory: bool = True):
    """
    Load FashionMNIST dataset with auto-download
    
    Args:
        batch_size: Batch size for dataloader
        resize: Optional resize size for images
        num_workers: Number of worker threads for dataloader
        pin_memory: Enable pin memory for faster transfer
    
    Returns:
        (train_loader, val_loader)
    """
    trans = [transforms.Grayscale(num_output_channels=3), 
             transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    
    # torchvision.datasets.FashionMNIST automatically downloads if not present
    print("📦 Loading FashionMNIST dataset...")
    mnist_train = torchvision.datasets.FashionMNIST(
        root=_project_root() / "dataset", train=True, transform=trans, download=True)
    mnist_val = torchvision.datasets.FashionMNIST(
        root=_project_root() / "dataset", train=False, transform=trans, download=True)
    
    print(f"✅ FashionMNIST loaded: {len(mnist_train)} train, {len(mnist_val)} val")
    
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        data.DataLoader(mnist_val, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    )