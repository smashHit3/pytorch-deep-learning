import torch
from torch import nn
import shared as d2l
import matplotlib.pyplot as plt

# LeNet-5
# 1. 输入层：32x32像素的灰度图像
# 2. 卷积层C1：6个5x5的卷积核，步幅为1，无填充，输出6x28x28的特征图
# 3. 激活函数：Sigmoid
# 4. 池化层S2：平均池化，窗口大小为2
# 5. 卷积层C3：16个5x5的卷积核，步幅为1，无填充，输出16x10x10的特征图
# 6. 激活函数：Sigmoid
# 7. 池化层S4：平均池化，窗口大小为2
# 8. 全连接层C5：将16x5x5的特征图展平为400维，连接到120个神经元
# 9. 激活函数：Sigmoid
# 10. 全连接层F6：连接120个神经元到84个神经元
# 11. 激活函数：Sigmoid
# 12. 输出层：连接84个神经元到10个输出，表示10个类别的概率
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), 
                    nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16 * 4 * 4, 120),
                    nn.Sigmoid(),
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    nn.Linear(84, 10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
lr, num_epochs = 0.9, 10

if __name__ == '__main__':
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()