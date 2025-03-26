import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

def compute_homography(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print(f"Computed homography matrix between {image1_path} and {image2_path}:")
    print(h)
    return h

def check_perspective_transform(homography_matrix, mask_shape):
    h, w = mask_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners = np.array([corners])
    transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)
    print(f"Original corners: {corners}")
    print(f"Transformed corners: {transformed_corners}")
    return transformed_corners

def test_homography_points(homography_matrix, mask_shape):
    h, w = mask_shape[:2]
    points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, homography_matrix)
    print(f"Original points: {points}")
    print(f"Transformed points: {transformed_points}")
    for i, pt in enumerate(transformed_points):
        if not (0 <= pt[0][0] < w and 0 <= pt[0][1] < h):
            print(f"Point {i} is out of bounds: {pt[0]}")
        else:
            print(f"Point {i} is within bounds: {pt[0]}")

def transform_mask(mask_path, homography_matrix, reference_image_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        raise ValueError(f"Failed to read reference image from {reference_image_path}")
    if mask is None:
        print(f"Failed to read mask from {mask_path}, using a black mask instead")
        mask = np.zeros((reference_image.shape[0], reference_image.shape[1]), dtype=np.uint8)
    else:
        if np.max(mask) == 0:
            print(f"Warning: Mask at {mask_path} is completely black")
        mask = cv2.resize(mask, (reference_image.shape[1], reference_image.shape[0]))
    h, w = reference_image.shape[:2]
    transformed_corners = check_perspective_transform(homography_matrix, mask.shape)
    test_homography_points(homography_matrix, mask.shape)
    transformed_mask = cv2.warpPerspective(mask, homography_matrix, (w, h))
    if np.max(transformed_mask) == 0:
        print(f"Warning: Transformed mask for {mask_path} resulted in a completely black image")
    else:
        print(f"Transformed mask contains valid pixels. Pixel range: min={np.min(transformed_mask)}, max={np.max(transformed_mask)}")
    cv2.imwrite('/path/to/debug/original_mask.png', mask)
    cv2.imwrite('/path/to/debug/warp_mask.png', transformed_mask)
    print(f"Saved original and warp masks for {mask_path}")
    return transformed_mask

def apply_threshold(mask, threshold=128):
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask

def calculate_maximum_intersection(masks):
    combined_mask = np.sum(np.array(masks), axis=0)
    max_intersection_value = np.max(combined_mask)
    max_intersection_mask = (combined_mask == max_intersection_value).astype(np.uint8) * 255
    return max_intersection_mask

def main():
    parser = argparse.ArgumentParser(description='Process images and masks for homography transformation.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask images')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output masks')
    args = parser.parse_args()

    image_dir = args.img_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    warp_output_dir = os.path.join(output_dir, 'warp_masks')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(warp_output_dir):
        os.makedirs(warp_output_dir)

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.jpg')],
        key=lambda x: int(x.split('.')[0])
    )

    for i in tqdm(range(len(image_files) - 1), desc="Processing images"):
        homographies = []
        for j in range(-3, 4):
            if i + j < 0 or i + j >= len(image_files):
                continue
            image1_path = os.path.join(image_dir, image_files[i+j])
            image2_path = os.path.join(image_dir, image_files[i])
            mask_path = os.path.join(mask_dir, f"{image_files[i+j].split('.')[0]}.png")
            homography_matrix = compute_homography(image1_path, image2_path)
            homographies.append((homography_matrix, mask_path, image_files[i+j]))
            print(f"Computed Homography Matrix for {image_files[i]} and {image_files[i+j]}")

        transformed_masks = []
        target_image_path = os.path.join(image_dir, image_files[i])
        for homography_matrix, mask_path, src_filename in homographies:
            warp_filename = f'{image_files[i].split(".")[0]}_from_{src_filename.split(".")[0]}.png'
            warp_output_path = os.path.join(warp_output_dir, warp_filename)
            transformed_mask = transform_mask(mask_path, homography_matrix, target_image_path)
            cv2.imwrite(warp_output_path, transformed_mask)
            print(f"Saved transformed mask: {warp_output_path}")
            transformed_masks.append(transformed_mask)

        max_intersection_mask = calculate_maximum_intersection(transformed_masks)
        final_mask = max_intersection_mask
        output_path = os.path.join(output_dir, f'{image_files[i].split(".")[0]}.jpg')
        cv2.imwrite(output_path, final_mask)
        if os.path.exists(output_path):
            print(f"Final combined mask saved as '{output_path}'")
        else:
            raise ValueError(f"Failed to write mask to: {output_path}")

if __name__ == "__main__":
    main()
