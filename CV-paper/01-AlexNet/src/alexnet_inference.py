import os
import json
import torch
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    return img_tensor


if __name__ == "__main__":
    """ =========== config =========== """
    # path_state_dict: 模型权重文件路径
    path_state_dict = os.path.join(BASE_DIR, "..", "data", "alexnet_state_dict.pth")
    # path_img: 待预测图片路径
    path_img = os.path.join(BASE_DIR, "..", "data", "predict", "GoldenRetrieverFromBaidu.jpg")
    # path_classnames: imagenet类别名称文件路径
    path_classnames = os.path.join(BASE_DIR, "..", "data", "predict", "imagenet1000.json")
    # path_classnames_cn: imagenet中文类别名称文件路径
    path_classnames_cn = os.path.join(BASE_DIR, "..", "data", "predict", "imagenet_classnames.txt")

    # print("path_state_dict: ", path_state_dict)
    # print("path_img: ", path_img)
    # print("path_classnames: ", path_classnames)
    # print("path_classnames_cn: ", path_classnames_cn)

    """ =========== load class names =========== """
    cls_n, cls_n_cn = load_classnames(path_classnames, path_classnames_cn)

    # print("cls_n: ", cls_n)
    # print("cls_n_cn: ", cls_n_cn)
    
