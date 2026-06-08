import torch
from torch import nn
import shared as d2l
import matplotlib.pyplot as plt

# AlexNet
# 1. 输入层：224x224像素的RGB图像
# 2. 卷积层1：96个11x11的卷积核，步幅为4，填充为1，输出96x55x55的特征图
# 3. 激活函数：ReLU
# 4. 池化层1：最大池化，窗口大小为3，步幅为2
# 5. 卷积层2：256个5x5的卷积核，步幅为1，填充为2，输出256x27x27的特征图
# 6. 激活函数：ReLU
# 7. 池化层2：最大池化，窗口大小为3，步幅为2
# 8. 卷积层3：384个3x3的卷积核，步幅为1，填充为1，输出384x13x13的特征图
# 9. 激活函数：ReLU
# 10. 卷积层4：384个3x3的卷积核，步幅为1，填充为1，输出384x13x13的特征图
# 11. 激活函数：ReLU
# 12. 卷积层5：256个3x3的卷积核，步幅为1，填充为1，输出256x13x13的特征图
# 13. 激活函数：ReLU
# 14. 池化层3：最大池化，窗口大小为3，步幅为2
# 15. 全连接层1：将256x6x6的特征图展平为9216维，连接到4096个神经元
# 16. 激活函数：ReLU
# 17. Dropout：丢弃率为0.5
# 18. 全连接层2：连接4096个神经元到4096个神经元
# 19. 激活函数：ReLU
# 20. Dropout：丢弃率为0.5
# 21. 输出层：连接4096个神经元到10个输出，表示10个类别的概率
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10))

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10

if __name__ == '__main__':
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()