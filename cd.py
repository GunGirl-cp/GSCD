import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from tqdm import tqdm
import cv2
from skimage.filters import threshold_otsu
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define deconvolution network for upsampling
class DeconvNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeconvNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_channels, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = nn.ReLU()(x)
        x = self.deconv2(x)
        x = nn.ReLU()(x)
        x = self.deconv3(x)
        x = nn.Sigmoid()(x)
        return x

# Load pretrained VGG16 (first 15 layers for feature extraction)
vgg16_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:15].to(device).eval()
print("VGG16 model loaded successfully.")

# Initialize the deconvolution network
deconv_net = DeconvNet(input_channels=256, output_channels=1).to(device)
print("DeconvNet initialized.")

# Preprocess image: resize and normalize
def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    print(f"Original image size: {orig_size}")
    image_resized = image.resize(target_size)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image_resized).unsqueeze(0)
    return image_tensor, orig_size

# Adaptive threshold and deconvolution upsampling using feature difference
def adaptive_threshold_and_deconv_upsample(feature_diff, orig_size):
    mask_up_tensor = deconv_net(feature_diff)
    print(f"DeconvNet output tensor shape: {mask_up_tensor.shape}")
    
    mask_up = mask_up_tensor.squeeze().cpu().detach().numpy()
    print(f"Mask after squeeze: {mask_up.shape}")

    thresh = threshold_otsu(mask_up)
    mask_up_binary = (mask_up > thresh).astype(np.uint8) * 255
    mask_up_resized = cv2.resize(mask_up_binary, orig_size, interpolation=cv2.INTER_NEAREST)
    mask_up_image = Image.fromarray(mask_up_resized)
    return mask_up_image

# Otsu-based change detection with an adjustment factor
def otsu_change_detection(img_path1, img_path2, adjust_factor=1.2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None:
        raise ValueError(f"Cannot read image: {img_path1}")
    if img2 is None:
        raise ValueError(f"Cannot read image: {img_path2}")
    
    if img1.shape != img2.shape:
        raise ValueError("Image dimensions do not match.")
    
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    ret, _ = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_threshold = min(ret * adjust_factor, 255)
    _, otsu_mask = cv2.threshold(gray_diff, adjusted_threshold, 255, cv2.THRESH_BINARY)
    
    print(f"[INFO] Otsu initial threshold: {ret:.2f}, adjusted threshold: {adjusted_threshold:.2f}")
    return otsu_mask

# Process folders and compute intersection of the two methods' results
def process_folders(before_folder, after_folder, output_folder, target_feature_size=(56,56), adjust_factor=1.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(before_folder) if f.lower().endswith(valid_ext)]
    
    for filename in tqdm(sorted(files), desc="Processing images"):
        before_image_path = os.path.join(before_folder, filename)
        after_image_path = os.path.join(after_folder, filename)
        
        if not os.path.exists(after_image_path):
            print(f"File {filename} not found in {after_folder}, skipping.")
            continue
        
        # Method 1: Feature-based change detection
        before_tensor, orig_size = preprocess_image(before_image_path, target_feature_size)
        after_tensor, _ = preprocess_image(after_image_path, target_feature_size)
        before_tensor, after_tensor = before_tensor.to(device), after_tensor.to(device)
        
        with torch.no_grad():
            features_before = vgg16_features(before_tensor)
            features_after = vgg16_features(after_tensor)
        feature_diff = torch.abs(features_before - features_after)
        print(f"Feature difference tensor shape: {feature_diff.shape}")
        mask_feature_pil = adaptive_threshold_and_deconv_upsample(feature_diff, orig_size)
        mask_feature = np.array(mask_feature_pil)
        
        # Method 2: Otsu-based change detection from second code
        mask_otsu = otsu_change_detection(before_image_path, after_image_path, adjust_factor)
        
        # Intersection of the two masks
        intersection_mask = cv2.bitwise_and(mask_feature, mask_otsu)
        
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, intersection_mask)
        print(f"Intersection mask saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change detection using feature difference and Otsu method with intersection")
    parser.add_argument('--before_folder', type=str, required=True, help="Path to the folder with before-change images")
    parser.add_argument('--after_folder', type=str, required=True, help="Path to the folder with after-change images")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save intersection masks")
    parser.add_argument('--adjust_factor', type=float, default=1.2, help="Threshold adjustment factor")
    args = parser.parse_args()
    
    process_folders(args.before_folder, args.after_folder, args.output_folder, target_feature_size=(56,56), adjust_factor=args.adjust_factor)
