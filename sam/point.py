import cv2
import numpy as np
import os

def load_image(image_path, gray=False):
    """加载图像，可选以灰度方式"""
    if gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    return image

def find_foreground_points_connected_components(mask):
    """在掩码的白色区域找到每个连通区域的质心"""
    # 使用 connectedComponentsWithStats 找到每个白色区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 提取所有前景点的质心
    foreground_points = []
    for i in range(1, num_labels):  # 从1开始跳过背景
        x, y, w, h, area = stats[i]
        if area > 0:  # 忽略面积为0的区域
            cX, cY = centroids[i]
            foreground_points.append([int(cX), int(cY)])

    return np.array(foreground_points)

def mark_foreground_points_on_mask(mask, points, color=(0, 0, 255), radius=5):
    """在掩码图像上标出前景点"""
    # 将灰度图像转换为BGR彩色图像，以便使用红色标记
    marked_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(marked_mask, tuple(point[::-1]), radius, color, -1)  # 注意point是(y, x)格式
    return marked_mask

def process_mask_with_connected_components(mask_path, output_path):
    """处理单张掩码图像，使用连通组件找到每个白色区域的质心并标记"""
    mask = load_image(mask_path, gray=True)

    foreground_points = find_foreground_points_connected_components(mask)

    if foreground_points.size > 0:
        marked_mask = mark_foreground_points_on_mask(mask, foreground_points)
    else:
        marked_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为BGR格式
        print("No white regions found in the mask.")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, marked_mask)
    print(f"Processed mask saved to {output_path}")

if __name__ == "__main__":
    mask_path = '/home/hello/lpf/4DGaussians/cd/ear/save/021.png'  # 替换为实际掩码路径
    output_path = '/home/hello/lpf/sam/masks.png'  # 替换为实际输出路径
    process_mask_with_connected_components(mask_path, output_path)
