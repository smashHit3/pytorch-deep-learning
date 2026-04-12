import torch
import torchvision
import shared as d2l
from torch import nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa5a736eebccfeb489cddc4cbd1e7a')
data_dir = d2l.download_extract('hotdog')
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
hotdogs = [train_imgs[i][0] for i in range(8)]
labels = [train_imgs.classes[train_imgs[i][1]] for i in range(8)]
d2l.show_images(hotdogs, 2, 4, titles=labels)
plt.show()