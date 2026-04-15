import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import shared as d2l
from PIL import Image
import numpy as np

# plt.figure(figsize=(16, 9))

# img = Image.open("img/cat1.jpg")
# img_array = np.array(img)

# plt.imshow(img_array)

# def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
#     Y = [aug(img) for _ in range(num_rows * num_cols)]
#     d2l.show_images(Y, num_rows, num_cols, scale=scale)

# # 1. Flip
# apply(img, torchvision.transforms.RandomHorizontalFlip())
# apply(img, torchvision.transforms.RandomVerticalFlip())

# shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)

# bright_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)
# apply(img, bright_aug)

# color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# augs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.RandomVerticalFlip(),
#     shape_aug,
#     bright_aug,
#     color_aug
# ])
# apply(img, augs)

all_images = torchvision.datasets.CIFAR10(train=True, root="../data", download=True)
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(train=is_train, root="../data", 
                                           download=True, transform=augs)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, 
                                       num_workers=d2l.get_dataloader_workers())

batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

if __name__ == "__main__":
    train_with_data_aug(train_augs, test_augs, net)
    plt.show()