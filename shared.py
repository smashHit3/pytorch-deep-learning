from torchvision import transforms
from torch.utils import data
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def use_svg_like_display():
    """Python 脚本中模拟 SVG 级高清显示"""
    # 使用支持高质量渲染的后端
    plt.switch_backend('TkAgg')  # 或 QtAgg

    # 关键：提高分辨率 + 抗锯齿，达到 SVG 级清晰度
    # plt.rcParams['figure.dpi'] = 300          # 超高DPI
    # plt.rcParams['savefig.dpi'] = 300
    # plt.rcParams['axes.linewidth'] = 0.8
    # plt.rcParams['font.size'] = 10
    plt.rcParams['lines.antialiased'] = True  # 开启抗锯齿

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(6, 4)):
        # 增加了legend参数
        if legend is None:
            legend = []
        use_svg_like_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 设置坐标轴
        self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.fmts = fmts
        self.legend = legend
        self.X, self.Y = None, None # 用来存储数据

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置坐标轴"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)

    def add(self, x, y):
        # 将新数据点添加到数据列表中
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            self.X[i].append(a)
            self.Y[i].append(b)
        # 重新绘制所有数据点
        self.axes[0].cla() # 清除当前坐标轴
        for x, y, fmt, label in zip(self.X, self.Y, self.fmts, self.legend):
            self.axes[0].plot(x, y, fmt, label=label) # 绘制数据点并加标签
        self.axes[0].legend() # 显示图例
        plt.draw() # 更新图形
        plt.pause(0.001) # 暂停以显示更新

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        # 创建一个长度为n的列表, 用于存储累加的结果, 初始值为0.0
        self.data = [0.0] * n

    def add(self, *args):
        # 将传入的参数与当前的累加结果相加, 并更新累加结果
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 将累加结果重置为0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        # 通过索引获取累加结果
        return self.data[idx]

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """开始计时"""
        self.tik = torch.cuda.Event(enable_timing=True)
        self.tok = torch.cuda.Event(enable_timing=True)
        self.tik.record()

    def stop(self):
        """结束计时"""
        self.tok.record()
        torch.cuda.synchronize() # 等待计时器停止
        self.times.append(self.tik.elapsed_time(self.tok))

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回时间的累积和"""
        return np.array(self.times).cumsum().tolist()

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    """w: 权重, b: 偏差, num_examples: 样本数量"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集, 然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, 
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, 
                            num_workers=get_dataloader_workers()))

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

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
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1] # 返回准确率

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期(定义见第3章)"""
    if isinstance(net, torch.nn.Module): 
        # 如果net是一个torch.nn.Module对象, 就将它设置为训练模式
        net.train()
    # 训练损失总和, 训练准确率总和, 样本数量
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward() # 反向传播计算梯度
            updater.step() # 更新参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward() # 反向传播计算梯度
            updater(X.shape[0]) # 更新参数, X.shape[0]是批量大小
        metric.add(float(l.sum().item()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型(定义见第3章)"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, f"训练损失过大: {train_loss}"
    assert train_acc < 1 and train_acc > 0.7, f"训练准确率过高或过低: {train_acc}"
    assert test_acc < 1 and test_acc > 0.7, f"测试准确率过高或过低: {test_acc}"

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 如果img是一个张量, 就将它转换为numpy数组
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def predict_ch3(net, test_iter, n=6): # 定义预测函数
    for X, y in test_iter: # 取一个小批量来预测
        break
    trues = get_fashion_mnist_labels(y) # 获取真实标签
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1)) # 获取预测标签
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)] # 将真实标签和预测标签拼接在一起作为标题
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]) # 显示前n张图片及其标题