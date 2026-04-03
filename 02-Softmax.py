import shared as d2l

import torch
from IPython import display
from matplotlib import pyplot as plt

def softmax(X): # 定义softmax函数
    X_exp = torch.exp(X) # 对每个元素求指数
    partition = X_exp.sum(1, keepdim=True) # 沿着行求和, 保持维度不变
    return X_exp / partition  # 这里应用了广播机制

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# 初始化权重参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def net(X): # 定义模型
    return softmax(torch.mm(X.view((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y): # 交叉熵损失函数
    return - torch.log(y_hat[range(len(y_hat)), y]) # 选择正确类别的概率的对数

def accuracy(y_hat, y): # 评估准确率
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: 
        # 如果y_hat是二维的, 就取每行最大的元素的索引作为预测类别
        y_hat = y_hat.argmax(axis=1) # 取每行最大元素的索引
    cmp = y_hat.type(y.dtype) == y # 比较预测类别和真实类别
    return float(cmp.type(y.dtype).sum()) # 计算正确预测的数量

def evaluate_accuracy(net, data_iter): # 评估模型在数据集上的准确率
    if isinstance(net, torch.nn.Module): 
        # 如果net是一个torch.nn.Module对象, 就将它设置为评估模式
        net.eval()  # 将模型设置为评估模式
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1] # 返回准确率

lr = 0.1

def updater(batch_size): # 定义优化算法
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10

def main():
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.predict_ch3(net, test_iter)
    plt.show()

if __name__ == '__main__':
    main()