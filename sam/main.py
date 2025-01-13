import sys
import numpy as np
import cv2
from tqdm import tqdm
import os
import glob
from segment_anything import sam_model_registry, SamPredictor

def load_image(image_path, gray=False):
    """加载图像，可选以灰度方式"""
    if gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    return image

def save_mask(mask, output_path):
    """将掩码保存为图像文件，并在保存前调整前景和背景色"""
    mask_uint8 = (mask * 255).astype(np.uint8)  # 将布尔类型的掩码转换为uint8类型

    # 如果掩码是三维的，而我们只需要一个通道，通常选择第一个通道
    if mask_uint8.ndim == 3:
        mask_uint8 = mask_uint8[0, :, :]  # 选择第一个通道

    # 减小图像尺寸
    height, width = mask_uint8.shape
    new_dimensions = (width // 2, height // 2)
    resized_mask = cv2.resize(mask_uint8, new_dimensions, interpolation=cv2.INTER_AREA)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, resized_mask)

def find_foreground_points(mask, area_threshold):
    """从掩码的每个白色区域提取一个前景点（质心）"""
    if mask.ndim > 2 and mask.shape[2] > 1:  # 检查是否是多通道图像
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    
    # 使用 connectedComponentsWithStats 找到每个白色区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 提取所有前景点的质心
    foreground_points = []
    for i in range(1, num_labels):  # 从1开始跳过背景
        x, y, w, h, area = stats[i]
        if area >= area_threshold:  # 忽略小区域
            cX, cY = centroids[i]
            foreground_points.append([int(cX), int(cY)])
    
    return np.array(foreground_points) if foreground_points else np.array([[0, 0]])

def process_directory(image_dir, mask_dir, output_dir, area_threshold, score_threshold=0.5):
    """处理目录中的所有图像和掩码对"""
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))  # 如果图像也是 PNG 格式
    sam_checkpoint = '/home/hello/lpf/sam/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for image_path in tqdm(image_files, desc="Processing images"):
        filename = os.path.basename(image_path)
        mask_filename = os.path.splitext(filename)[0] + '.png'  # 将掩码文件名扩展名改为 JPG
        mask_path = os.path.join(mask_dir, mask_filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Processing {filename}")

        if os.path.exists(mask_path):
            image = load_image(image_path)
            mask = load_image(mask_path, gray=True)

            # 确保掩码的大小与图像大小相同
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            print(f"Loaded image and mask for {filename}, image shape: {image.shape}, mask shape: {mask.shape}")

            # 将图像和掩码调整为原始大小的一半
            image_height, image_width = image.shape[:2]
            new_dimensions = (image_width // 2, image_height // 2)
            resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(mask, new_dimensions, interpolation=cv2.INTER_AREA)

            foreground_points = find_foreground_points(resized_mask, area_threshold)
            print(f"{filename}: {len(foreground_points)} foreground points")
            input_labels = np.ones(foreground_points.shape[0], dtype=np.int32)

            if len(foreground_points) > 0 and not np.array_equal(foreground_points, [[0, 0]]):
                predictor.set_image(resized_image)
                masks_list = []
                for point in foreground_points:
                    masks, scores, logits = predictor.predict(
                        point_coords=point.reshape(1, -1),
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=True
                    )
                    # 过滤掉低置信度的掩码
                    for mask, score in zip(masks, scores):
                        if score > score_threshold:
                            masks_list.append(mask)

                # 检查是否有掩码被添加到列表中
                if masks_list:
                    # 将所有掩码进行逻辑或操作
                    final_mask = np.logical_or.reduce(masks_list)
                    save_mask(final_mask, output_path)
                    print(f"Saved mask for {filename} to {output_path}")
                else:
                    final_mask = np.zeros_like(mask)
                    save_mask(final_mask, output_path)
                    print(f"No masks with score above threshold for {filename}, saving a black mask")
            else:
                print(f"No valid foreground points found for {filename}")
        else:
            print(f"Mask not found for {filename}")

if __name__ == "__main__":
    image_dir = '/home/hello/lpf/camera'
    mask_dir = '/home/hello/lpf/goodcamera'
    output_dir = '/home/hello/lpf/sam/result/camera'
    area_threshold = 0  # 设置面积阈值
    score_threshold = 0.7 # 设置分数阈值
    process_directory(image_dir, mask_dir, output_dir, area_threshold, score_threshold)
