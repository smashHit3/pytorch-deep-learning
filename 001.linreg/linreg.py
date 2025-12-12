import os
import random
import torch
import matplotlib
# If no X display is available (headless), use the non-interactive Agg backend.
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    # 1. 生成特征矩阵 X
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 从正态分布 N(0, 1) 生成随机数
    # 形状: [样本数, 特征数]

    # 2. 计算无噪声的标签 y
    y = torch.matmul(X, w) + b
    #   torch.matmul(X, w): 矩阵乘法 X × w
    #   结果形状: [样本数, 1]

    # 3. 添加高斯噪声
    y += torch.normal(0, 0.01, y.shape)
    #   添加 N(0, 0.01²) 的噪声

    # 4. 返回结果，确保 y 是列向量
    return X, y.reshape((-1, 1))
    #   reshape((-1, 1)): 确保 y 形状为 [样本数, 1]

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    # 1. 获取样本总数
    num_examples = len(features)  # features.shape[0]
    
    # 2. 创建索引列表 [0, 1, 2, ..., num_examples-1]
    indices = list(range(num_examples))
    
    # 3. 随机打乱索引顺序（重要！）
    random.shuffle(indices)  # Shuffle the data
    
    # 4. 循环生成批次
    for i in range(0, num_examples, batch_size):
        # 4.1 获取当前批次的索引切片
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
            # min() 处理最后一批可能不足 batch_size 的情况
        
        # 4.2 使用 yield 返回当前批次（生成器）
        yield features[batch_indices], labels[batch_indices]
        # yield 使函数成为生成器，每次迭代返回一个批次

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """Mean squared error loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save    
    """Minibatch stochastic gradient descent."""
    # 1. 进入无梯度计算模式
    with torch.no_grad():
        # 2. 遍历所有参数（如权重w、偏置b）
        for param in params:
            # 3. 参数更新：param = param - lr * gradient / batch_size
            param -= lr * param.grad / batch_size
            # 关键：除以 batch_size，因为损失是批次的平均损失
            
            # 4. 清零梯度（重要！）
            param.grad.zero_()
            # 否则梯度会累积，导致更新错误

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):# 外层循环：遍历整个数据集多次
    for X, y in data_iter(batch_size, features, labels):# 内层循环：遍历每个小批次
        # 1. 前向传播计算损失
        l = loss(net(X, w, b), y)  # Compute loss
        # 2. 反向传播计算梯度
        l.sum().backward()          # Backpropagate
        # 3. 优化器更新参数
        sgd([w, b], lr, batch_size) # Update parameters
    # 每个epoch结束后：评估整个训练集
    with torch.no_grad():# 禁用梯度计算，节省内存
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    # 假设 loss() 返回每个样本的损失（形状: [batch_size, 1]）
    """l = loss(net(X, w, b), y)"""      # 形状: [32, 1]，32个样本各自的损失

    # 方法A：先求和再反向传播（代码中的方式）
    """l.sum().backward()"""             # 计算总损失的梯度
    # 梯度 = ∂(Σloss_i)/∂θ = Σ(∂loss_i/∂θ)

    # 方法B：先平均再反向传播
    """l.mean().backward()"""            # 计算平均损失的梯度  
    # 梯度 = ∂(mean(loss_i))/∂θ = (1/batch_size) × Σ(∂loss_i/∂θ)

    # 在SGD更新时：param -= lr * grad / batch_size
    # 两种方法最终效果等价，因为：
    # 方法A: grad_total = Σ(∂loss_i/∂θ), 更新时除以batch_size
    # 方法B: grad_mean = (1/b) × Σ(∂loss_i/∂θ), 更新时不除以batch_size

print('true_w:', true_w, '\nestimated_w:', w.reshape(true_w.shape))
print('true_b:', true_b, '\nestimated_b:', b)

print('features:', features[0],"\nlabel:", labels[0])
# Ensure tensors are on CPU and detached before converting to NumPy
x = features[:, 1].cpu().detach().numpy()
y = labels.cpu().detach().numpy()

plt.figure(figsize=(16, 9))
plt.scatter(x, y, s=1)
# If headless, save to file; otherwise show interactively
if os.environ.get('DISPLAY', '') == '':
    out = 'scatter.png'
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')
else:
    plt.show()
