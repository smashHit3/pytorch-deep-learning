"""
@File: alexnet_inference.py
@Description: 使用训练好的AlexNet模型进行图像分类推理
@Author: QianHua Liu
@Email: 1983561291@qq.com
@Date: 2026-04-15
"""
import json
import torch
import tools.config as config
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

def load_classnames(path_classnames, path_classnames_cn):
    """ 加载类别名称
    Args:
        path_classnames: imagenet类别名称文件路径 json文件, 每行一个类别名称, 共1000行
        path_classnames_cn: imagenet中文类别名称文件路径 txt文件, 每行一个中文类别名称, 共1000行
    Returns:
        classnames: imagenet类别名称列表
        classnames_cn: imagenet中文类别名称列表
    """
    with open(path_classnames, "r") as f:
        # json.load()函数将JSON格式的字符串解析为Python对象，这里是一个列表，包含了1000个类别名称
        classnames = json.load(f)
    with open(path_classnames_cn, "r", encoding="utf-8") as f:
        # 逐行读取中文类别名称，并用strip()去除每行的前后空白字符
        classnames_cn = [line.strip() for line in f]

    return classnames, classnames_cn

def process_img(path_img):
    """ 处理输入图像
    Args:
        path_img: 待预测图片路径
    Returns:
        img_tensor: 处理后的图像张量，形状为[      1,     3, 224, 224]
                                        [批次大小, 通道数, 高度, 宽度]
        img_rgb: 处理前的RGB图像对象, PIL.Image.Image类型
    """
    inference_transform = transforms.Compose([
        transforms.Resize(256),  # 将输入图像调整为256x256
        transforms.CenterCrop(224),  # 从中心裁剪出224x224的图
        transforms.ToTensor(),  # 将图像转换为PyTorch张量，并将像素值归一化到[0, 1]范围
        transforms.Normalize(mean=config.norm_mean, 
                             std=config.norm_std)  # 使用ImageNet的均值和标准差进行归一化
    ])
    img_rgb = Image.open(path_img).convert("RGB")  # 打开图像并转换为RGB模式
    img_tensor = inference_transform(img_rgb)  # 对图像进行预处理
    img_tensor = img_tensor.unsqueeze(0)  # 在第0维添加一个批次维
    img_tensor = img_tensor.to(config.device)  # 将图像张量移动到设备（CPU或GPU）
    return img_tensor, img_rgb

if __name__ == "__main__":
    """ =========== print config =========== """
    print("path_img: ", config.path_img)
    print("path_state_dict: ", config.path_state_dict)
    print("path_classnames: ", config.path_classnames)
    print("path_classnames_cn: ", config.path_classnames_cn)

    """ =========== load class names =========== """
    classnames, classnames_cn = load_classnames(config.path_classnames, config.path_classnames_cn)

    print("cls_n.shape: ", len(classnames))
    print("cls_n_cn.shape: ", len(classnames_cn))

    """ =========== process image =========== """
    img_tensor, img_rgb = process_img(config.path_img)

    print("img_tensor shape: ", img_tensor.shape)
    print("img_rgb: ", img_rgb)

    """ =========== load model =========== """
    alexnet_model = config.load_model(config.path_state_dict, vis_model=True)

    print("alexnet_model: ", alexnet_model)

    """ =========== inference =========== """
    with torch.no_grad():  # 在推理过程中禁用梯度计算，以节省内存和加速计算
        time_start = torch.cuda.Event(enable_timing=True)  # 创建一个CUDA事件对象，用于记录推理开始时间
        time_end = torch.cuda.Event(enable_timing=True)  # 创建另一个CUDA事件对象，用于记录推理结束时间
        time_start.record()  # 记录推理开始时间
        output = alexnet_model(img_tensor)  # 将处理后的图像张量输入模型进行推理
        time_end.record()  # 记录推理结束时间
        torch.cuda.synchronize()  # 等待所有CUDA操作完成，以确保时间测量的准确性
        inference_time = time_start.elapsed_time(time_end)  # 计算推理时间，单位为毫秒
        print(f"Inference time: {inference_time:.2f} ms")
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # 对模型输出的logits进行softmax，得到每个类别的概率分布
        top5_prob, top5_catid = torch.topk(probabilities, 5)  # 获取概率最高的前5个类别的概率值和对应的类别索引
    
    print("Top-5 predicted categories and probabilities:")
    for i in range(top5_prob.size(0)):
        print(f"{classnames_cn[top5_catid[i]]} ({classnames[top5_catid[i]]}): {top5_prob[i].item():.4f}")

    """ =========== visualize results =========== """
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)  # 显示输入图像
    plt.title(f"Predicted: {classnames_cn[top5_catid[0]]} ({classnames[top5_catid[0]]})\nInference time: {inference_time:.2f} ms")  # 设置标题，显示预测的类别和推理时间
    plt.axis("off")  # 关闭坐标轴显示
    plt.show()  # 显示图像
