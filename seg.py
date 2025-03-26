import sys
import numpy as np
import cv2
from tqdm import tqdm
import os
import glob
from segment_anything import sam_model_registry, SamPredictor
import argparse

def load_image(image_path, gray=False):
    """Load an image, optionally in grayscale."""
    if gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
    return image

def save_mask(mask, output_path):
    """Save the mask as an image file after adjusting its scale and size."""
    mask_uint8 = (mask * 255).astype(np.uint8)  # Convert boolean mask to uint8
    if mask_uint8.ndim == 3:
        mask_uint8 = mask_uint8[0, :, :]  # Use the first channel if mask is multi-channel
    height, width = mask_uint8.shape
    new_dimensions = (width // 2, height // 2)
    resized_mask = cv2.resize(mask_uint8, new_dimensions, interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized_mask)

def find_foreground_points(mask, area_threshold):
    """Extract a foreground point (centroid) from each white region in the mask."""
    if mask.ndim > 2 and mask.shape[2] > 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    foreground_points = []
    for i in range(1, num_labels):  # Skip background label 0
        x, y, w, h, area = stats[i]
        if area >= area_threshold:
            cX, cY = centroids[i]
            foreground_points.append([int(cX), int(cY)])
    return np.array(foreground_points) if foreground_points else np.array([[0, 0]])

def process_directory(image_dir, mask_dir, output_dir, area_threshold, score_threshold=0.5):
    """Process all image-mask pairs in the specified directories."""
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    sam_checkpoint = '/home/hello/lpf/sam/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for image_path in tqdm(image_files, desc="Processing images"):
        filename = os.path.basename(image_path)
        mask_filename = os.path.splitext(filename)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Processing {filename}")

        if os.path.exists(mask_path):
            image = load_image(image_path)
            mask = load_image(mask_path, gray=True)

            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            print(f"Loaded image and mask for {filename}, image shape: {image.shape}, mask shape: {mask.shape}")

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
                    for mask_val, score in zip(masks, scores):
                        if score > score_threshold:
                            masks_list.append(mask_val)
                if masks_list:
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
    parser = argparse.ArgumentParser(
        description="SAM-based segmentation with mask adjustment and foreground point extraction"
    )
    parser.add_argument('--image_dir', type=str, required=True, help="Directory of input images")
    parser.add_argument('--mask_dir', type=str, required=True, help="Directory of input masks")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output masks")
    parser.add_argument('--area_threshold', type=int, default=0, help="Area threshold for foreground points")
    parser.add_argument('--score_threshold', type=float, default=0.7, help="Score threshold for mask selection")
    args = parser.parse_args()

    process_directory(args.image_dir, args.mask_dir, args.output_dir, args.area_threshold, args.score_threshold)
