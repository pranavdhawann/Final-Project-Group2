import numpy as np
import cv2
import torch

def gaussian2D(shape, sigma=1):
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

def draw_gaussian(heatmap, center, radius):
    x, y = center
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=radius / 3)

    x, y = int(x), int(y)
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_h = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if masked_h.shape == masked_g.shape:
        np.maximum(masked_h, masked_g, out=masked_h)

def create_heatmap(boxes, image_shape, output_size=180):
    heatmap = np.zeros((1, output_size, output_size), dtype=np.float32)
    size_map = np.zeros((2, output_size, output_size), dtype=np.float32)
    offset_map = np.zeros((2, output_size, output_size), dtype=np.float32)
    mask = np.zeros((output_size, output_size), dtype=np.float32)

    down_ratio = image_shape[0] // output_size

    for box in boxes:
        x, y, w, h = box["bbox"]
        cx, cy = x + w / 2, y + h / 2
        cx_out, cy_out = int(cx / down_ratio), int(cy / down_ratio)

        radius = max(1, int(min(w, h) / down_ratio / 2))
        draw_gaussian(heatmap[0], (cx_out, cy_out), radius)

        size_map[:, cy_out, cx_out] = [w, h]
        offset_map[:, cy_out, cx_out] = [cx / down_ratio - cx_out, cy / down_ratio - cy_out]
        mask[cy_out, cx_out] = 1

    return heatmap, size_map, offset_map, mask