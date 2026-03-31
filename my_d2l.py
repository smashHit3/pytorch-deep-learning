import torch

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    """w: 权重, b: 偏差, num_examples: 样本数量"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
