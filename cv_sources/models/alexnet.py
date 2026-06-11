"""
AlexNet implementation
@File: alex_net.py
@Description: AlexNet模型定义
"""

import torch
import torch.nn as nn

MODEL_TYPE_ALEXNET = "alexnet"

class AlexNet(nn.Module):
    def __init__(self, num_classes : int = 1000, dropout : float = 0.5, init_weights=False) -> None:
        super().__init__()
        self.feasures = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 卷积层1
            nn.ReLU(inplace=True), # 激活函数1
            nn.MaxPool2d(kernel_size=3, stride=2), # 池化层1
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # 卷积层2
            nn.ReLU(inplace=True), # 激活函数2
            nn.MaxPool2d(kernel_size=3, stride=2), # 池化层2
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 卷积层3
            nn.ReLU(inplace=True), # 激活函数3
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 卷积层4
            nn.ReLU(inplace=True), # 激活函数4
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 卷积层5
            nn.ReLU(inplace=True), # 激活函数5
            nn.MaxPool2d(kernel_size=3, stride=2), # 池化层3
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # 自适应平均池化层
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), # Dropout层1
            nn.Linear(256 * 6 * 6, 4096), # 全连接
            nn.ReLU(inplace=True), # 激活函数6
            nn.Dropout(p=dropout), # Dropout层2
            nn.Linear(4096, 4096), # 全连接
            nn.ReLU(inplace=True), # 激活函数7
            nn.Linear(4096, num_classes), # 全连接
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.feasures(x) # 通过卷积层提取特征
        x = self.avgpool(x) # 通过自适应平均池化层调整特征图大小
        x = torch.flatten(x, 1) # 将特征图展平为向量
        x = self.classifier(x) # 通过全连接层进行分类
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
