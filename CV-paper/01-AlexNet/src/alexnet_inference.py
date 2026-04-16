import os
import json
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision import models
from torchsummary import summary
from PIL import Image

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
# 获取当前文件所在目录的绝对路径，作为后续文件路径的基准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 检查是否有可用的GPU，如果有则使用GPU进行计算，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差进行归一化
    ])
    img_rgb = Image.open(path_img).convert("RGB")  # 打开图像并转换为RGB模式
    img_tensor = inference_transform(img_rgb)  # 对图像进行预处理
    img_tensor = img_tensor.unsqueeze(0)  # 在第0维添加一个批次维
    img_tensor = img_tensor.to(device)  # 将图像张量移动到设备（CPU或GPU）
    return img_tensor, img_rgb

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

if __name__ == "__main__":
    """ =========== config =========== """
    # path_img: 待预测图片路径
    path_img = os.path.join(BASE_DIR, "..", "data", "predict", "GoldenRetrieverFromBaidu.jpg")
    # path_state_dict: 模型权重文件路径
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet_state_dict.pth")
    # path_classnames: imagenet类别名称文件路径
    path_classnames = os.path.join(BASE_DIR, "..", "data", "predict", "imagenet1000.json")
    # path_classnames_cn: imagenet中文类别名称文件路径
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "predict", "imagenet_classnames.txt")

    print("path_img: ", path_img)
    print("path_state_dict: ", path_state_dict)
    print("path_classnames: ", path_classnames)
    print("path_classnames_cn: ", path_classnames_cn)

    """ =========== load class names =========== """
    cls_n, cls_n_cn = load_classnames(path_classnames, path_classnames_cn)

    print("cls_n.shape: ", len(cls_n))
    print("cls_n_cn.shape: ", len(cls_n_cn))
    
    """ =========== process image =========== """
    img_tensor, img_rgb = process_img(path_img)

    print("img_tensor shape: ", img_tensor.shape)
    print("img_rgb: ", img_rgb)

    """ =========== load model =========== """
    alexnet_model = load_model(path_state_dict, vis_model=True)

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
        print(f"{cls_n_cn[top5_catid[i]]} ({cls_n[top5_catid[i]]}): {top5_prob[i].item():.4f}")

    """ =========== visualize results =========== """
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)  # 显示输入图像
    plt.title(f"Predicted: {cls_n_cn[top5_catid[0]]} ({cls_n[top5_catid[0]]})\nInference time: {inference_time:.2f} ms")  # 设置标题，显示预测的类别和推理时间
    plt.axis("off")  # 关闭坐标轴显示
    plt.show()  # 显示图像
