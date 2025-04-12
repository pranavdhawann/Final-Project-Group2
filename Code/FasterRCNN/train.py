import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
import cv2
from utilities import get_frcnn_annotations
from custom_dataset import CustomDataset

model = fasterrcnn_resnet50_fpn(weights=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

transforms = T.Compose([
    T.ToTensor(),
])

dataset = CustomDataset(transforms=transforms)
indices_random = torch.randperm(len(dataset)).tolist()

train_size = int(len(dataset) * 0.7)

train_dataset = torch.utils.data.Subset(dataset, indices_random[:train_size])
val_dataset = torch.utils.data.Subset(dataset, indices_random[train_size:])

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn= lambda batch: tuple(zip(*batch))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle= False,
    collate_fn= lambda batch: tuple(zip(*batch))
)


device = torch.device('cuda') if torch.cuda.is_available()\
                              else torch.device('cpu')

model.to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    trainable_params,
    lr= 0.005,
    momentum=0.9,
    weight_decay=5e-4,
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 1
for epoch in range(num_epochs):

    model.train()
    train_loss= 0

    for images, targets in train_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    lr_scheduler.step()
    print(f'Epoch: {epoch+1}, Loss: {train_loss/len(train_loader):.4f}')

print('training done')


model.eval()

with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        predictions = model(images)

        print(predictions)


#TODO
"""
- inference.py
- save and load model
- print val_loss on each epoch 
"""





