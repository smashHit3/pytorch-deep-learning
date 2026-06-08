import torchvision
from torchvision import transforms
from torch.utils import data
from pathlib import Path

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    print(f"dataset path: {_project_root() / "dataset"}")
    
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=_project_root() / "dataset", train=True, transform=trans, download=True)
    mnist_val = torchvision.datasets.FashionMNIST(
        root=_project_root() / "dataset", train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_val, batch_size, shuffle=False, num_workers=4))