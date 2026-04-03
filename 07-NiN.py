import torch
from torch import nn
import shared as d2l

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), 
        nn.ReLU(), 
        nn.Conv2d(in_channels, out_channels, n=1), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.ReLU())

