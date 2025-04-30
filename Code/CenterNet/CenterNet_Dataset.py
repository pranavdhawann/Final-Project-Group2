import torch
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from centernet.utils import create_heatmap

class MotorDataset(Dataset):
    def __init__(self, img_dir, ann_path, transform=None):
        self.img_dir = img_dir
        self.data = json.load(open(ann_path))
        self.imgs = self.data["images"]
        self.anns = self.data["annotations"]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_info = self.imgs[idx]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        image = np.expand_dims(image, 0)

        boxes = [ann for ann in self.anns if ann["image_id"] == img_info["id"]]
        heatmap, size_map, offset_map, mask = create_heatmap(boxes, image.shape[1:])

        return {
            "image": torch.FloatTensor(image),
            "heatmap": torch.FloatTensor(heatmap),
            "size": torch.FloatTensor(size_map),
            "offset": torch.FloatTensor(offset_map),
            "mask": torch.FloatTensor(mask)
        }