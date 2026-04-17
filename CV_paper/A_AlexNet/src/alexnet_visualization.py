import os

from torch import nn
from PIL import Image
from tools import config
from torchvision import transforms
from torchvision import utils as vtuils
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    log_dir = os.path.join(config.results_dir, "alexnet_visualization")
    writer = SummaryWriter(log_dir=log_dir, filename_suffix='_kernel')

    alexnet_model = config.load_model(config.path_state_dict, vis_model=True)

    kernel_num = -1
    vis_max_num = 1
    for sub_model in alexnet_model.modules():
        if not isinstance(sub_model, nn.Conv2d):
            continue
        
        kernel_num += 1
        if kernel_num > vis_max_num:
            break
        
        kernels = sub_model.weight
        print(f"conv{kernel_num} kernels shape: {kernels.shape}")
        channels_out, channels_in, kernel_height, kernel_width = kernels.shape
        for i in range(channels_out):
            kernel = kernels[i].unsqueeze(1)  # 选择第i个卷积核，形状为[channels_in, kernel_height, kernel_width]
            kernel_grid = vtuils.make_grid(kernel, nrow=8, normalize=True, scale_each=True)
            writer.add_image(f"conv{kernel_num}_kernel_{i}", kernel_grid, global_step=i)

        kernels_all = kernels.view(-1, 3, kernel_height, kernel_width)
        kernels_grid = vtuils.make_grid(kernels_all, nrow=8, normalize=True, scale_each=True)
        writer.add_image(f"conv{kernel_num}_kernels_all", kernels_grid, global_step=0)

    writer.close()
    writer = SummaryWriter(log_dir=log_dir, filename_suffix='_feature_map')

    train_norm_mean = [0.49139968, 0.48215827, 0.44653124]
    train_norm_std = [0.24703233, 0.24348505, 0.26158768]
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_norm_mean, std=train_norm_std)
    ])
    
    image_rgb = Image.open(config.path_img).convert("RGB")
    image_tensor = img_transform(image_rgb).unsqueeze(0).to(config.device)

    conv1_feature_map = alexnet_model.features[0](image_tensor).transpose(1, 0)
    conv1_grid = vtuils.make_grid(conv1_feature_map, nrow=8, normalize=True, scale_each=True)
    writer.add_image("conv1_feature_map", conv1_grid, global_step=0)
    print(f"conv1 feature map shape: {conv1_feature_map.shape}")
    writer.close()