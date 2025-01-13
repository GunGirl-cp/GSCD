import os
import glob
import shutil
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.filters import threshold_otsu
import numpy as np
import cv2 as cv
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保输出目录存在
output_folder = "/home/hello/lpf/4DGaussians/cd/pencil/mask"
os.makedirs(output_folder, exist_ok=True)

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
def generate_mask(img1_path, img2_path, output_path):
    img1, original_size1 = preprocess_image(img1_path)
    img2, original_size2 = preprocess_image(img2_path)

    img1, img2 = img1.to(device), img2.to(device)

    features1 = extract_features(img1, conv_layers)
    features2 = extract_features(img2, conv_layers)

    feature_diffs = [torch.abs(f1 - f2) for f1, f2 in zip(features1, features2)]
    
    last_diff = feature_diffs[-1][0, :, :, :].data.cpu().numpy()
    mask = np.mean(last_diff, axis=0)



    #thresh = threshold_otsu(mask) * 2     #potting
    #thresh = 20   #desk
    # thresh = 10    #lego
    #thresh = 18    #windowsill
    # thresh = 18
    thresh = 25
    # thresh = 12
    #thresh = 12


    binary_mask = (mask > thresh).astype(np.uint8) * 255
    
    binary_mask_resized = cv.resize(binary_mask, original_size1, interpolation=cv.INTER_NEAREST)

    
    mask_img = Image.fromarray(binary_mask_resized)
    mask_img.save(output_path)
    print(f"Saved difference mask to {output_path}")

# 复制并重命名图像以创建 img1 和 img2 文件夹
def copy_and_rename_images(source_folder, destination_folder1, destination_folder2):
    # 确保目标文件夹存在
    os.makedirs(destination_folder1, exist_ok=True)
    os.makedirs(destination_folder2, exist_ok=True)
    
    # 获取源文件夹中的所有文件
    files = sorted(os.listdir(source_folder))
    
    # 计算阈值，作为源文件数量的一半
    threshold = len(files) // 2

    for file in files:
        filename, extension = os.path.splitext(file)
        if filename.isdigit():
            file_number = int(filename)
            source_path = os.path.join(source_folder, file)
            if file_number >= threshold:
                new_filename = f"{file_number - threshold:03d}{extension}"
                destination_path = os.path.join(destination_folder2, new_filename)
                shutil.copy(source_path, destination_path)
            else:
                new_filename = f"{file_number:03d}{extension}"
                destination_path = os.path.join(destination_folder1, new_filename)
                shutil.copy(source_path, destination_path)

# 定义路径
base_path = '/home/hello/lpf/4DGaussians/cd/pencil'
source_folder = '/home/hello/lpf/4DGaussians/output/llff/pencil/train/ours_60000/renders'
destination_folder1 = os.path.join(base_path, 'img1')
destination_folder2 = os.path.join(base_path, 'img2')
output_folder_maskgnn = os.path.join(base_path, 'maskgnn')
save_path = os.path.join(base_path, 'save')
com_path = os.path.join(base_path, 'com')
output_folder_mask = os.path.join(base_path, 'mask')


#复制并重命名图像
copy_and_rename_images(source_folder, destination_folder1, destination_folder2)

# 确保输出目录存在
for folder in [output_folder_maskgnn, save_path, com_path, output_folder_mask]:
    os.makedirs(folder, exist_ok=True)

# VGG 生成 mask
before_image_folder = destination_folder1
after_image_folder = destination_folder2
for before_image_path in tqdm(sorted(glob.glob(os.path.join(before_image_folder, '*.png'))), desc="生成 VGG mask"):
    image_name = os.path.basename(before_image_path)
    after_image_path = os.path.join(after_image_folder, image_name)
    
    if os.path.exists(after_image_path):
        output_path = os.path.join(output_folder_maskgnn, image_name.replace('.png', '.png'))
        generate_mask(before_image_path, after_image_path, output_path)
    else:
        print(f"未找到与 {image_name} 对应的后图像")

# 阈值法生成 mask
image_files = [file for file in os.listdir(destination_folder1) if file.endswith('.JPG') or file.endswith('.png')]
number_of_images = len(image_files)
threshold_value = 120# 根据需要调整阈值

threshold_value = 40#windowsill
threshold_value = 20#windowsill


def change_detection(image1, image2, threshold):
    img1 = cv.imread(image1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(image2, cv.IMREAD_GRAYSCALE)
    if img1.shape != img2.shape:
        img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv.absdiff(img1, img2)
    _, binary_diff = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
    return binary_diff

for i in range(number_of_images):
    i_str = str(i).zfill(3)
    image1_path = os.path.join(destination_folder1, f'{i_str}.png')
    image2_path = os.path.join(destination_folder2, f'{i_str}.png')
    result = change_detection(image1_path, image2_path, threshold_value)
    cv.imwrite(os.path.join(save_path, f'{i_str}.png'), result)

# 生成交集 mask
def intersection_of_masks(mask_path1, mask_path2, output_path):
    mask1 = Image.open(mask_path1).convert('L')
    mask2 = Image.open(mask_path2).convert('L')
    if mask1.size != mask2.size:
        raise ValueError("掩码图像必须具有相同的尺寸")
    intersection_img = Image.new('L', mask1.size)
    for x in range(mask1.width):
        for y in range(mask1.height):
            pixel1 = mask1.getpixel((x, y))
            pixel2 = mask2.getpixel((x, y))
            intersection_img.putpixel((x, y), 255 if pixel1 == 255 and pixel2 == 255 else 0)
    intersection_img.save(output_path)

save_masks = glob.glob(os.path.join(save_path, '*.png'))
maskgnn_masks = glob.glob(os.path.join(output_folder_maskgnn, '*.png'))
maskgnn_dict = {os.path.basename(path).replace('.png', '.png'): path for path in maskgnn_masks}

for save_mask_path in save_masks:
    image_name = os.path.basename(save_mask_path)
    maskgnn_mask_path = maskgnn_dict.get(image_name)
    
    if maskgnn_mask_path:
        output_path = os.path.join(output_folder_mask, image_name)
        intersection_of_masks(save_mask_path, maskgnn_mask_path, output_path)
        print(f"交集 mask 已保存到: {output_path}")
    else:
        print(f"未找到与 {image_name} 对应的 maskgnn 图像")

# 显示对比结果
for i in range(number_of_images):
    i_str = str(i).zfill(3)
    img1 = cv.imread(os.path.join(destination_folder1, f'{i_str}.png'))
    img2 = cv.imread(os.path.join(destination_folder2, f'{i_str}.png'))
    mask = cv.imread(os.path.join(save_path, f'{i_str}.png'))
    cv.imwrite(os.path.join(com_path, f'{i_str}-1.png'), img1)
    cv.imwrite(os.path.join(com_path, f'{i_str}-3.png'), img2)
    cv.imwrite(os.path.join(com_path, f'{i_str}-2.png'), mask)