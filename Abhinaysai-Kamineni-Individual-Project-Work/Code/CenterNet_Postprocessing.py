"""
postprocessing_and_augmentation.py
Demonstrates:
  1) Saliency map and Grad-CAM for CenterNet outputs
  2) Bounding-box augmentation using Albumentations
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip,
    Rotate, GaussNoise, Blur, BboxParams
)
from CenterNet_Dataset import MotorDataset, IMG_W, IMG_H
from CenterNet_Model import CenterNetFPN as CenterNet, DEVICE

# ─── Saliency Map ─────────────────────────────────────
def generate_saliency_map(model, input_tensor):
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True).to(DEVICE)
    hm, _, _, _ = model(input_tensor)
    score = hm.max()
    model.zero_grad()
    score.backward()
    saliency = input_tensor.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

# ─── Grad-CAM ─────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval().to(DEVICE)
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor):
        input_tensor = input_tensor.to(DEVICE)
        hm, _, _, _ = self.model(input_tensor)
        score = hm.max()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # Smooth for better visualization
        cam = cv2.GaussianBlur((cam * 255).astype(np.uint8), (3, 3), 0) / 255.0
        return cam

# ─── Visualization ────────────────────────────────────
def visualize_cam_on_image(image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ─── Albumentations bbox augmentation ─────────────────
def get_albumentations_transform():
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.5),
        Blur(blur_limit=3, p=0.3),
        GaussNoise(p=0.3),
    ], bbox_params=BboxParams(
        format='pascal_voc',            # use absolute [x_min, y_min, x_max, y_max]
        label_fields=['labels'],
        min_visibility=0.3
    ))

def augment_with_albumentations(image, bboxes, labels):
    H, W = image.shape[:2]
    # Clamp to image bounds
    clamped = []
    for x1, y1, x2, y2 in bboxes:
        cx1, cy1 = np.clip(x1, 0, W - 1), np.clip(y1, 0, H - 1)
        cx2, cy2 = np.clip(x2, 0, W - 1), np.clip(y2, 0, H - 1)
        if cx2 > cx1 and cy2 > cy1:
            clamped.append([cx1, cy1, cx2, cy2])
    if not clamped:
        return image, bboxes, labels

    transform = get_albumentations_transform()
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image
    data = transform(image=rgb, bboxes=clamped, labels=labels)

    aug_img = data['image']
    aug_labels = data['labels']
    aug_bboxes = []
    for (nx1, ny1, nx2, ny2), _lbl in zip(data['bboxes'], aug_labels):
        x1 = int(np.clip(nx1, 0, W))
        y1 = int(np.clip(ny1, 0, H))
        x2 = int(np.clip(nx2, 0, W))
        y2 = int(np.clip(ny2, 0, H))
        if x2 > x1 and y2 > y1:
            aug_bboxes.append([x1, y1, x2, y2])
    if not aug_bboxes:
        return image, clamped, labels
    return aug_img, aug_bboxes, aug_labels

# ─── MAIN DEMO ────────────────────────────────────────
if __name__ == '__main__':
    ds = MotorDataset(debug=False)
    model = CenterNet().to(DEVICE)

    # Hook a deeper layer for Grad-CAM
    target_layer = model.layer4[-1].conv3
    cam_gen = GradCAM(model, target_layer)

    # Augmentation demo
    rec = ds.samples[0]
    img0 = cv2.imread(rec['path'], cv2.IMREAD_GRAYSCALE)
    x, y, w, h = rec['boxes'][0]
    bboxes = [[x, y, x + w, y + h]]
    labels = [1]
    aug_img, aug_bboxes, _ = augment_with_albumentations(img0, bboxes, labels)
    side = np.hstack([
        cv2.cvtColor(cv2.resize(img0, (IMG_W, IMG_H)), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.resize(aug_img, (IMG_W, IMG_H)), cv2.COLOR_RGB2BGR)
    ])
    cv2.imwrite('orig_vs_aug.png', side)
    print('Saved orig_vs_aug.png')

    # Grad-CAM demo
    img_t, _, _ = ds[0]
    input_tensor = img_t.unsqueeze(0).to(DEVICE)
    cam = cam_gen(input_tensor)
    gray = (img_t[0].cpu().numpy() * 255).astype(np.uint8)
    overlay = visualize_cam_on_image(gray, cam)
    cv2.imwrite('gradcam_overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print('Saved gradcam_overlay.png')