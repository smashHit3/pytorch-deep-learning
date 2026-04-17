"""
@FileName: config.py
@Description: 配置文件
@Author: QianHua Liu
@Email: 1983561291@qq.com
@Date: 2026-04-16
"""
import os
import torch
from torchvision import models
from torchsummary import summary

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "..", "data")
results_dir = os.path.join(base_dir, "..", "..", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"

path_img = os.path.join(data_dir, "predict", "GoldenRetrieverFromBaidu.jpg")
path_state_dict = os.path.join(data_dir, "alexnet_state_dict.pth")
path_classnames = os.path.join(data_dir, "predict", "imagenet1000.json")
path_classnames_cn = os.path.join(data_dir, "predict", "imagenet_classnames.txt")

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

max_epochs = 100
batch_size = 256
learning_rate = 0.001
log_interval = 10
validation_interval = 1
lr_decay_step = 1
num_workers = 4
freeze_layer_flag = True

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

def load_model(path_state_dict, vis_model=False):
    """ 加载AlexNet模型并加载预训练权重
    Args:
        path_state_dict: 模型权重文件路径
        vis_model: 是否可视化模型结构, 默认为False
    Returns:
        model: 加载了预训练权重的AlexNet模型
    """
    model = models.alexnet()  # 加载AlexNet模型结构，但不加载预训练权重
    pretrained_state_dict = torch.load(path_state_dict, weights_only=False)  # 加载预训练模型权重
    model.load_state_dict(pretrained_state_dict)  # 将预训练权重加载到模型中
    model = model.to(device)  # 将模型移动到设备（CPU或GPU）
    model.eval()  # 将模型设置为评估模式
    if vis_model:
        summary(model, (3, 224, 224))  # 可视化模型结构，输入尺寸为(3, 224, 224)
    return model
