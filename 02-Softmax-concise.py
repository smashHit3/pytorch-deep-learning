import matplotlib.pyplot as plt
import shared as d2l

import torch
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义模型
# 这里我们使用了一个简单的全连接层来实现softmax回归模型
# 该层包含784个输入和10个输出, 因为每张图片有28*28=784个像素, 而Fashion-MNIST数据集有10个类别
# nn.Sequential是一个容器, 可以将多个层组合在一起, 这里我们先使用nn.Flatten()将输入的图片展平为一维向量, 然后使用nn.Linear()定义一个全连接层
# nn.Linear(784, 10)表示输入维度为784, 输出维度为10的全连接层
# 通过这种方式定义模型, 我们就不需要手动定义权重参数W和b了, 因为nn.Linear会自动创建这些参数并进行初始化
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    # 这里我们定义了一个初始化函数, 用于初始化模型的权重参数
    # 该函数接受一个模块m作为输入, 如果该模块是一个全连接层(nn.Linear), 就使用正态分布来初始化它的权重参数, 标准差为0.01
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 在训练模型之前, 我们需要对模型的权重参数进行初始化
# 这里我们使用了net.apply(init_weights)来对模型中的每个模块都应用init_weights函数, 从而完成权重参数的初始化
net.apply(init_weights)

# 定义损失函数, 这里我们使用了交叉熵损失函数, reduction='none'表示不对损失进行平均或求和, 而是返回每个样本的损失值
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1) # 定义优化算法, 这里我们使用了随机梯度下降(SGD)算法, 学习率为0.1

num_epochs = 10

def main():
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.predict_ch3(net, test_iter)
    plt.show()

if __name__ == "__main__":
    main()