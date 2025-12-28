import os
import cv2
import numpy as np

def parse_label_file(label_path):
    polygons = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # First value is class, rest are x1 y1 x2 y2 ...
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polygons.append(polygon)
    return polygons

def convert_yolov8seg_to_masks(image_dir, label_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            print(f'No label for {img_file}, skipping.')
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f'Could not read image {img_file}, skipping.')
            continue
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = parse_label_file(label_path)
        for poly in polygons:
            pts = np.array([[int(x * width), int(y * height)] for x, y in poly], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + '.png')
        cv2.imwrite(mask_path, mask)
        print(f'Saved mask for {img_file} -> {mask_path}')

if __name__ == '__main__':
    # For train
    convert_yolov8seg_to_masks(
        'data/train/images',
        'data/train/labels',
        'data/train/masks')