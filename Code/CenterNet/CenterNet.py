import os
import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


# ---------------------- SET RANDOM SEED ----------------------
def set_seed(seed=420):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(420)


# ---------------------- UTILITY FUNCTIONS ----------------------
def gaussian2D(shape, sigma=1):
    """Generate a 2D Gaussian kernel."""
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius):
    """Draw a 2D Gaussian on the heatmap at the specified center and radius."""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if masked_gaussian.shape != masked_heatmap.shape:
        return heatmap
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def create_heatmap(bbox_list, image_size, output_size=180):
    heatmap = np.zeros((output_size, output_size), dtype=np.float32)
    size_map = np.zeros((2, output_size, output_size), dtype=np.float32)
    offset_map = np.zeros((2, output_size, output_size), dtype=np.float32)
    mask = np.zeros((output_size, output_size), dtype=np.float32)
    down_ratio = image_size[0] // output_size  # assume square image

    for bbox in bbox_list:
        x, y, w, h = bbox["bbox"]
        x_center = x + w / 2.0
        y_center = y + h / 2.0
        x_center_out = x_center / down_ratio
        y_center_out = y_center / down_ratio
        x_int = min(int(x_center_out), output_size - 1)  # clamp index to valid range
        y_int = min(int(y_center_out), output_size - 1)
        offset_x = x_center_out - x_int
        offset_y = y_center_out - y_int
        radius = max(1, int(min(w, h) / down_ratio / 2))
        heatmap = draw_gaussian(heatmap, (x_center_out, y_center_out), radius)
        size_map[:, y_int, x_int] = [w, h]
        offset_map[:, y_int, x_int] = [offset_x, offset_y]
        mask[y_int, x_int] = 1

    heatmap = np.expand_dims(heatmap, axis=0)  # shape: (1, H, W)
    return heatmap, size_map, offset_map, mask


# ---------------------- LOSS FUNCTIONS ----------------------
def focal_loss(pred, gt, alpha=2, beta=4):
    pos_inds = (gt == 1).float()
    neg_inds = (gt < 1).float()
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    pos_loss = -torch.log(pred) * ((1 - pred) ** alpha) * pos_inds
    neg_loss = -torch.log(1 - pred) * (pred ** alpha) * ((1 - gt) ** beta) * neg_inds
    num_pos = pos_inds.sum()
    loss = (pos_loss + neg_loss).sum() / (num_pos + 1e-6)
    return loss


def l1_loss(pred, target, mask):
    return torch.sum(torch.abs(pred * mask - target * mask)) / (torch.sum(mask) + 1e-6)


# ---------------------- MODEL DEFINITION ----------------------
class CenterNet(nn.Module):
    def __init__(self):
        super(CenterNet, self).__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.hm_head = self._make_head(1)
        self.wh_head = self._make_head(2)
        self.off_head = self._make_head(2)

    def _make_head(self, out_channels):
        return nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        pred_hm = torch.sigmoid(self.hm_head(feat))
        pred_wh = self.wh_head(feat)
        pred_off = self.off_head(feat)
        # Upsample predictions (e.g., from 23x23 to 180x180)
        pred_hm = F.interpolate(pred_hm, size=(180, 180), mode='bilinear', align_corners=False)
        pred_wh = F.interpolate(pred_wh, size=(180, 180), mode='bilinear', align_corners=False)
        pred_off = F.interpolate(pred_off, size=(180, 180), mode='bilinear', align_corners=False)
        return {"heatmap": pred_hm, "size": pred_wh, "offset": pred_off}


# ---------------------- DATASET DEFINITION ----------------------
class MotorDataset(Dataset):
    def __init__(self, img_dir, csv_path):
        self.img_dir = os.path.abspath(img_dir)
        self.labels = pd.read_csv(csv_path)
        self.samples = []
        for tomo_id in self.labels['tomo_id'].unique():
            subfolder = os.path.join(self.img_dir, tomo_id)
            if not os.path.isdir(subfolder):
                print(f"Warning: No folder found for tomo_id: {tomo_id}")
                continue
            # Find files containing "slice" (case-insensitive)
            slice_files = sorted([
                f for f in os.listdir(subfolder)
                if "slice" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
            ])
            if not slice_files:
                slice_files = sorted([
                    f for f in os.listdir(subfolder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
                ])
                if slice_files:
                    print(f"Warning: No file with 'slice' in name for tomo_id {tomo_id}. Using {slice_files[0]}")
                else:
                    print(f"Warning: No image files found for tomo_id: {tomo_id}")
                    continue
            row = self.labels[self.labels['tomo_id'] == tomo_id].iloc[0]
            x_center = row["Motor axis 1"]
            y_center = row["Motor axis 2"]
            bbox = [x_center - 50, y_center - 50, 100, 100]
            for sfile in slice_files:
                full_path = os.path.join(subfolder, sfile)
                self.samples.append({
                    "tomo_id": tomo_id,
                    "slice_file": full_path,
                    "bbox": bbox
                })
        print(f"Dataset loaded with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        image_path = sample_info["slice_file"]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            folder = os.path.dirname(image_path)
            available = os.listdir(folder)
            raise FileNotFoundError(f"Image not found for tomo_id: {sample_info['tomo_id']} at {image_path}. "
                                    f"Available files: {available}")
        image = image.astype(np.float32) / 255.0
        image = cv2.resize(image, (720, 720))
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)  # Convert 1 channel to 3 channels
        bbox_list = [{"bbox": sample_info["bbox"]}]
        heatmap, size_map, offset_map, mask = create_heatmap(bbox_list, (720, 720))
        if idx == 0:
            print(f"Loaded sample 0: tomo_id = {sample_info['tomo_id']}, slice file = {sample_info['slice_file']}")
        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "heatmap": torch.tensor(heatmap, dtype=torch.float32),
            "size": torch.tensor(size_map, dtype=torch.float32),
            "offset": torch.tensor(offset_map, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "tomo_id": sample_info["tomo_id"],
            "slice_file": sample_info["slice_file"]
        }


# ---------------------- TRAINING PIPELINE ----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Increase batch size and number of workers to improve throughput
    dataset = MotorDataset("../../Dataset/train", "../../Dataset/train_labels.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    model = CenterNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    # Enable mixed precision training using AMP for faster training
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            images = batch["image"].to(device)
            gt_heatmap = batch["heatmap"].to(device)
            gt_size = batch["size"].to(device)
            gt_offset = batch["offset"].to(device)
            mask = batch["mask"].to(device).unsqueeze(1)  # (B,1,H,W)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss_hm = focal_loss(outputs["heatmap"], gt_heatmap)
                loss_size = l1_loss(outputs["size"], gt_size, mask)
                loss_offset = l1_loss(outputs["offset"], gt_offset, mask)
                loss = loss_hm + 0.1 * loss_size + loss_offset
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), "centernet_model.pth")
    print("Model saved as centernet_model.pth")


if __name__ == "__main__":
    train()