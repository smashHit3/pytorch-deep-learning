import shared as d2l

import torch
import numpy as np
from torch.utils import data
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # 将数据数组包装成TensorDataset对象
    # 返回一个DataLoader对象, 用于迭代数据, batch_size指定每个批次的大小, shuffle指定是否打乱数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

for X, y in data_iter:
    print(X, '\n', y)
    break

# 定义模型
net = nn.Sequential(nn.Linear(2, 1)) # 2个输入特征, 1个输出特征

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 权重参数初始化为均值为0、标准差为0.01的正态分布随机数
net[0].bias.data.fill_(0) # 偏差参数初始化为0

# 定义损失函数
loss = nn.MSELoss() # 均方误差损失函数(Mean Squared Error Loss)

# 定义优化算法
# 随机梯度下降优化算法, net.parameters()返回模型的所有参数, lr指定学习率
trainer = torch.optim.SGD(net.parameters(), lr=0.03) 

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y.view(-1, 1)) # 计算损失
        trainer.zero_grad() # 梯度清零
        l.backward()    # 反向传播计算梯度
        trainer.step()  # 更新参数
    with torch.no_grad():
        train_l = loss(net(features), labels.view(-1, 1)) # 计算训练损失
        print(f'epoch {epoch + 1}, loss {float(train_l):f}')

print('true_w:', true_w, '\nlearned w:', net[0].weight.data)
print('true_b:', true_b, '\nlearned b:', net[0].bias.data)