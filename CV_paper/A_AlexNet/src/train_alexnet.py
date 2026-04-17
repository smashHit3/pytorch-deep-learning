"""
@FileName: train_alexnet.py
@Description: 训练AlexNet模型
@Author: QianHua Liu
@Email: 1983561291@qq.com
@Date: 2026-04-16
"""
import os
import torch
import tools.config as config
import matplotlib.pyplot as plt

from tools.cat_dog_dataset import CatDogDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def ten_crop_to_tensor_and_normalize(crops):
    return torch.stack([transforms.Normalize(mean=config.norm_mean, std=config.norm_std)(transforms.ToTensor()(crop)) for crop in crops])

if __name__ == "__main__":
    # 定义训练数据的变换，包括调整图像大小、中心裁剪、随机裁剪、随机水平翻转、转换为张量和归一化
    train_transform = transforms.Compose([
        # 将输入图像调整为256x256
        transforms.Resize((256)),
        # 从中心裁剪出256x256的图像
        transforms.CenterCrop(256),
        # 随机裁剪出224x224的图像
        transforms.RandomCrop(224),
        # 随机水平翻转图像，概率为0.5
        transforms.RandomHorizontalFlip(p=0.5),
        # 将图像转换为PyTorch张量，并将像素值归一化到[0, 1]范围
        transforms.ToTensor(),
        # 使用ImageNet的均值和标准差进行归一化
        transforms.Normalize(mean=config.norm_mean, 
                             std=config.norm_std)
    ])

    # 定义验证数据的变换，包括调整图像大小、中心裁剪、十裁剪、转换为张量和归一化
    validation_transform = transforms.Compose([
        # 将输入图像调整为256x256
        transforms.Resize((256, 256)),
        # 对图像进行十裁剪，生成10个224x224的裁剪图像，vertical_flip=False表示不进行垂直翻转
        transforms.TenCrop(224, vertical_flip=False),
        # 将每个裁剪图像转换为张量，并使用ImageNet的均值和标准差进行归一化，最后将10个裁剪图像堆叠成一个4D张量
        transforms.Lambda(ten_crop_to_tensor_and_normalize)
    ])

    # 创建训练和验证数据集，并使用DataLoader加载数据，设置批量大小、是否打乱数据和使用的线程数
    train_dataset = CatDogDataset(data_dir=config.train_dir, mode="train", transform=train_transform)
    validation_dataset = CatDogDataset(data_dir=config.train_dir, mode="val", transform=validation_transform)

    # 使用DataLoader加载训练和验证数据集，设置批量大小、是否打乱数据和使用的线程数
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              shuffle=True, num_workers=config.num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, 
                                   shuffle=False, num_workers=config.num_workers)

    # 加载预训练的AlexNet模型，并将其移动到指定的设备（CPU或GPU）
    alexnet_model = config.load_model(config.path_state_dict, vis_model=True)

    # 定义损失函数为交叉熵损失函数，适用于多分类问题
    loss_fn = torch.nn.CrossEntropyLoss()

    # 冻结卷积层的参数，使其在训练过程中不更新
    if config.freeze_layer_flag:
        fc_parameters = list(alexnet_model.classifier.parameters())  # 获取全连接层的参数
        fc_parameter_ids = list(map(id, fc_parameters))  # 获取全连接层参数的ID列表
        base_parameters = [p for p in alexnet_model.parameters() if id(p) not in fc_parameter_ids]  # 获取卷积层的参数
        optimizer = torch.optim.SGD([
            {"params": base_parameters, "lr": config.learning_rate * 0.1},  # 卷积层使用较小的学习率
            {"params": fc_parameters, "lr": config.learning_rate}  # 全连接层使用原始学习率
        ], momentum=0.9)
    else:
        # 如果不冻结卷积层，则所有参数都使用相同的学习率进行优化
        optimizer = torch.optim.SGD(alexnet_model.parameters(), lr=config.learning_rate, momentum=0.9)

    # 定义学习率调度器，每隔一定的训练步骤（step_size）将学习率乘以一个衰减因子（gamma），以便在训练过程中逐渐降低学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=0.1)

    for epoch in range(config.max_epochs):
        train_loss = 0.0 # 累积训练损失
        train_correct = 0.0 # 累积正确预测的样本数量
        train_total = 0.0 # 累积训练样本总数

        alexnet_model.train() # 将模型设置为训练模式，以启用dropout和batch normalization等训练特定的行为
        # 遍历训练数据加载器中的每个批次，获取输入图像和对应的标签，并将它们移动到指定的设备（CPU或GPU）
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 将输入图像和标签移动到指定的设备（CPU或GPU），以便在训练过程中进行计算
            images, labels = images.to(config.device), labels.to(config.device)
            # 在训练过程中，首先将优化器的梯度清零，以避免累积之前的梯度，然后将输入图像传递给模型进行前向传播，计算损失函数的值，并进行反向传播以计算梯度，最后更新模型的参数
            optimizer.zero_grad()
            outputs = alexnet_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # 累积训练损失，并计算当前批次的预测结果，更新正确预测的样本数量和总样本数量
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % config.log_interval == 0:
                print(f"Epoch [{epoch + 1}/{config.max_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {train_loss / (batch_idx + 1):.4f}, "
                      f"Accuracy: {100 * train_correct / train_total:.2f}%")
        # 每个epoch结束后，调用学习率调度器的step()方法更新学习率，根据预设的衰减策略调整学习率，以便在训练过程中逐渐降低学习率，提高模型的收敛性能
        scheduler.step()
        if epoch % config.validation_interval == 0:
            validation_loss = 0.0 # 累积验证损失
            validation_correct = 0.0 # 累积正确预测的样本数量
            validation_total = 0.0 # 累积验证样本总数

            alexnet_model.eval() # 将模型设置为评估模式，以禁用dropout和batch normalization等训练特定的行为
            with torch.no_grad():
                for images, labels in validation_loader:
                    # 将输入图像和标签移动到指定的设备（CPU或GPU），以便在验证过程中进行计算
                    images, labels = images.to(config.device), labels.to(config.device)

                    # 获取输入图像的批次大小、裁剪数量、通道数、高度和宽度等维度信息，以便后续处理
                    batch_size, num_crops, channels, height, width = images.size()
                    # 将输入的图像张量从形状(batch_size, num_crops, channels, height, width)
                    # 调整为(batch_size * num_crops, channels, height, width)，以适应模型的输入要求
                    images = images.view(-1, channels, height, width)
                    outputs = alexnet_model(images)
                    # 将模型的输出从形状(batch_size * num_crops, num_classes)调整为(batch_size, num_crops, num_classes)
                    outputs = outputs.view(batch_size, num_crops, -1)
                    # 对每个样本的多个裁剪结果取平均，得到最终的预测结果
                    outputs = outputs.mean(dim=1)
                    loss = loss_fn(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    validation_total += labels.size(0)
                    validation_correct += (predicted == labels).sum().item()
                    validation_loss += loss.item()
                
            print(f"Epoch [{epoch + 1}/{config.max_epochs}], Validation Loss: {validation_loss / len(validation_loader):.4f}, "
                  f"Validation Accuracy: {100 * validation_correct / validation_total:.2f}%")
            alexnet_model.train()

    # 训练完成后保存模型
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "alexnet_trained.pth")
    torch.save(alexnet_model.state_dict(), save_path)
    print(f"训练完成，模型已保存到: {save_path}")