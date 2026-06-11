import torchvision
from torchvision import transforms
from torch.utils import data
from pathlib import Path

DATASET_NAME_FASHION_MNIST = "fashion_mnist"
NUM_CLASSES = 10


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_data_fashion_mnist(batch_size: int, resize: int=None):
    trans = [transforms.Grayscale(num_output_channels=3), 
             transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=_project_root() / "dataset", train=True, transform=trans, download=True)
    mnist_val = torchvision.datasets.FashionMNIST(
        root=_project_root() / "dataset", train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_val, batch_size, shuffle=False, num_workers=4))