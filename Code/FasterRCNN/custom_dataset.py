import torch
from utilities import get_frcnn_annotations, save_image, preprocessImg
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np

get_train_transform=  A.Compose([
        # Basic augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #
        # # Color transforms
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),

        # Geometric transforms
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=0.05,
            rotate=(-30, 30),
            shear=(-5, 5),
            p=0.5
        ),
        #
        # # Image quality
        # A.OneOf([
        #     A.Blur(blur_limit=3, p=0.3),
        #     A.MedianBlur(blur_limit=3, p=0.3),
        # ], p=0.2),
        #
        # # Convert to tensor
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        # min_visibility=0.2,
        label_fields=['labels']
    ))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", transforms=None):
        self.transforms = transforms
        self.data_path = "/home/ubuntu/Final-Project-Group2/Dataset/"
        self.img_resize_shape = (900, 900)
        self.annotations = get_frcnn_annotations(self.data_path + "train_labels_split.csv", self.img_resize_shape)
        self.annotations = [x for x in self.annotations if x["split"] == split]

    def __getitem__(self, idx):
        img_path = self.data_path + "train/" + self.annotations[idx]["file_name"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = preprocessImg(img)
        # Get boxes and labels
        boxes = self.annotations[idx]["boxes"]
        labels = self.annotations[idx]["labels"]
        boxes = boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes
        labels = labels.tolist() if isinstance(labels, torch.Tensor) else labels
        if self.transforms is not None:
            try:
                transformed = self.transforms(
                    image=img,
                    bboxes=boxes,
                    labels=labels
                )
                img = transformed['image']
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            except Exception as e:
                print(f"Error transforming image {img_path}: {e}")
                # Return empty tensors if transformation fails
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
        else:
            img = ToTensorV2()(image=img)['image']
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Handle empty boxes
        if boxes.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }
        img = img.float() / 255.0
        return img, target

    def __len__(self):
        return len(self.annotations)


if __name__ == "__main__":
    dataset = CustomDataset("train", get_train_transform)

    for i in range(100):
        idx = random.randint(0, len(dataset) - 1)
        try:
            img, target = dataset[idx]
            save_image(img, target, f"./viz_bboxes/output_{i}.jpg")
        except Exception as e:
            print(f"Error processing image {idx}: {e}")