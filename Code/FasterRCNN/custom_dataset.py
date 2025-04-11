import torch
import torchvision.transforms as T
import cv2
from utilities import get_frcnn_annotations, save_image

transform = T.Compose([
    T.ToTensor(),
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.data_path = "/home/ubuntu/Final-Project-Group2/Dataset/"
        self.img_resize_shape = (720, 720)
        self.annotations = get_frcnn_annotations(self.data_path+"train_labels.csv", self.img_resize_shape)

    def __getitem__(self, idx):

        img_path = self.data_path + "train/"+self.annotations[idx]["file_name"]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_resize_shape)

        boxes = self.annotations[idx]["boxes"]
        labels = self.annotations[idx]["labels"]

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    def __len__(self):
        return len(self.annotations)


if __name__ == "__main__":

    dataset = CustomDataset(transform)
    import random

    for i in range(20):
        img, target = dataset[random.randint(0, len(dataset) - 1)]

        save_image(img, target, f"./viz_bboxes/output_{i}.jpg")

