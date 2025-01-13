import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2 as cv
from PIL import Image
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的VGG16模型
try:
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features[:15].to(device).eval()
except RuntimeError as e:
    print(f"CUDA 内存不足，切换到 CPU 进行计算：{e}")
    device = torch.device("cpu")
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features[:15].to(device).eval()

# 冻结模型的所有参数
for param in model.parameters():
    param.requires_grad = False

# 提取卷积层及其权重
conv_layers = [layer for layer in model if isinstance(layer, nn.Conv2d)]
print(f"Total convolutional layers: {len(conv_layers)}")

# 图像预处理函数
def preprocess_image(img_path):
    img = cv.imread(img_path)
    original_size = (img.shape[1], img.shape[0])  # 存储原始图像大小 (width, height)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = np.array(img)
    img = transform(img)
    img = img.unsqueeze(0)
    return img, original_size

# 通过所有卷积层传递图像
def extract_features(img, conv_layers):
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    return results

# 生成掩码图像
def generate_mask(img1_path, img2_path, output_folder, manual_threshold=10):
    img1, original_size1 = preprocess_image(img1_path)
    img2, original_size2 = preprocess_image(img2_path)

    img1, img2 = img1.to(device), img2.to(device)

    features1 = extract_features(img1, conv_layers)
    features2 = extract_features(img2, conv_layers)

    feature_diffs = [torch.abs(f1 - f2) for f1, f2 in zip(features1, features2)]
    
    last_diff = feature_diffs[-1][0, :, :, :].data.cpu().numpy()
    mask = np.mean(last_diff, axis=0)
    
    # 使用手动设定的阈值
    binary_mask = (mask > manual_threshold).astype(np.uint8) * 255
    
    binary_mask_resized = cv.resize(binary_mask, original_size1, interpolation=cv.INTER_NEAREST)
    
    mask_img = Image.fromarray(binary_mask_resized)
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 构造输出文件路径
    base_filename = os.path.basename(img1_path)
    name, ext = os.path.splitext(base_filename)
    output_path = os.path.join(output_folder, f"{name}_threshold_{manual_threshold}{ext}")
    
    mask_img.save(output_path)
    print(f"Saved difference mask to {output_path}")

# 示例用法
img1_path = '/home/hello/lpf/4DGaussians/cd/windowsill/img1/000.png'
img2_path = '/home/hello/lpf/4DGaussians/cd/windowsill/img2/000.png'
output_folder = '/home/hello/lpf/4DGaussians/findy/windowsill'

# 尝试不同的阈值
for threshold in [1, 5, 10, 15, 20, 30, 40]:
    print(f"Trying threshold: {threshold}")
    generate_mask(img1_path, img2_path, output_folder, manual_threshold=threshold)
