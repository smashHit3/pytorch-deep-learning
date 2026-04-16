"""
@File: CatDogDataset.py
@Description: 自定义数据集类，用于加载猫狗图像数据
@Author: QianHua Liu
@Email: 1983561291@qq.com
@Date: 2026-04-16
"""
import os
import random
from PIL import Image

class CatDogDataset:
    def __init__(self, data_dir, mode="train", split_ratio=0.8, random_seed=42, transform=None):
        """ 初始化数据集
        Args:
            data_dir: 数据目录，包含猫狗图像的文件夹路径
            mode: 数据集模式，"train"表示训练集，"val"表示验证集
            split_ratio: 训练集和验证集的划分比例, 默认为0.8, 即80%训练集, 20%验证集
            random_seed: 随机种子，用于保证划分的可重复性
            transform: 数据增强和预处理的变换函数, 默认为None
        """
        self.data_dir = data_dir
        self.mode = mode
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.transform = transform

        # 加载数据并进行划分
        self.image_paths, self.labels = self.load_data()

    def __getitem__(self, index):
        """ 根据索引获取图像路径和标签
        Args:
            index: 数据索引
        Returns:
            image: 图像对象
            label: 图像标签, 1表示狗, 0表示猫
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            # 如果定义了数据变换函数，则对图像进行变换
            image = self.transform(image)

        return image, label

    def __len__(self):
        if len(self.image_paths) == 0:
            raise ValueError("No images found in the specified directory.")
        """ 返回数据集的大小，即图像数量 """
        return len(self.image_paths)

    def load_data(self):
        """ 加载数据并进行划分
        Returns:
            image_paths: 划分后的图像路径列表
            labels: 划分后的标签列表, 1表示狗, 0表示猫
        """
        img_names = os.listdir(self.data_dir)  # 获取数据目录下的所有文件名
        random.seed(self.random_seed)  # 设置随机种子，保证划分的可重复性
        random.shuffle(img_names)  # 打乱文件名列表，增加数据的随机性

        img_labels = [1 if "dog" in name else 0 for name in img_names]  # 假设文件名中包含"dog"表示狗，否则表示猫

        split_index = int(len(img_names) * self.split_ratio)  # 计算划分索引
        if self.mode == "train":
            selected_img_names = img_names[:split_index]  # 选择训练集的文件名
            selected_labels = img_labels[:split_index]  # 选择训练集的标签
        elif self.mode == "val":
            selected_img_names = img_names[split_index:]  # 选择验证集的文件名
            selected_labels = img_labels[split_index:]  # 选择验证集的标签
        else:
            raise ValueError("mode must be 'train' or 'val'")
        
        image_paths = [os.path.join(self.data_dir, name) for name in selected_img_names]  # 构建图像路径列表
        return image_paths, selected_labels