import torch
import matplotlib.pyplot as plt
import numpy as np
import random

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    """w: 权重, b: 偏差, num_examples: 样本数量"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])

plt.figure(figsize=(6, 4))  # Use pyplot.figure to set the figure size
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
# 保存图片（支持 png/jpg/pdf/svg）
plt.savefig("my_plot.png")
# 必须加：关闭画布，释放内存
plt.close()