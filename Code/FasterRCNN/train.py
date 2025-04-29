import torch
from scipy.stats.tests.test_continuous_fit_censored import optimizer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utilities import validate, saveResultImages, save_metrics_plots
from params import *
import argparse
import os


from custom_dataset import CustomDataset

model = fasterrcnn_resnet50_fpn(
    pretrained=True,
    box_score_thresh=0.2,
    box_nms_thresh=0.3,
    rpn_pre_nms_top_n_train=1000,
    rpn_post_nms_top_n_train=500,
)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

train_transforms = T.Compose([
    T.ToTensor(),
])

val_transforms = T.Compose([
    T.ToTensor(),
])

import albumentations as A
from albumentations.pytorch import ToTensorV2


get_train_transform= A.Compose([
    A.HorizontalFlip(p=0.3),  # Reduced from 0.5
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.1),  # Add noise
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),  # New
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

get_val_transform= A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))


train_dataset = CustomDataset(transforms=get_train_transform)
val_dataset = CustomDataset(split= "val",transforms=get_val_transform)

print("Len of train_dataset: ", len(train_dataset))
print("Len of val_dataset: ", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn= lambda batch: tuple(zip(*batch)),
    num_workers= 4,
    pin_memory= True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle= False,
    collate_fn= lambda batch: tuple(zip(*batch)),
    num_workers= 4,
    pin_memory= True
)


device = torch.device('cuda') if torch.cuda.is_available()\
                              else torch.device('cpu')

model.to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5  # Lower minimum learning rate
)

def train(run_name):
    patience_counter = 0
    best_map50 = -1
    train_losses = []
    val_losses = []
    precisions = []
    recalls = []
    f1s = []
    for epoch in range(EPOCHS):

        model.train()
        train_loss= 0
        val_loss = 0
        for images, targets in train_loader:

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        # lr_scheduler.step()
        scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader))
        map_ = validate(model, val_loader, device)

        print(f'Epoch: {epoch+1}, train_loss: {train_loss/len(train_loader):.4f}, val_loss: {val_loss/len(val_loader):.4f}')
        print(f"     map: {map_["map"]:.4f} map50: {map_['map_50']:.4f} map75: {map_['map_75']:.4f} p: {map_['precision']:.4f} r: {map_['recall']:.4f} f1: {map_['f1']:.4f}")
        precisions.append(map_['precision'])
        recalls.append(map_['recall'])
        f1s.append(map_['f1'])
        if map_['map_50'] > best_map50:
            patience_counter = 0
            best_map50 = map_['map_50']
            torch.save(model.state_dict(), f"runs/{run_name}/best_models/best_model.pth")
            print("Model Saved")
        else:
            patience_counter +=1

        if patience_counter == PATIENCE:
            print("Early Stopping")
            break
    final_map = validate(model, val_loader, device)
    print(f"     map: {final_map["map"]:.4f} map50: {final_map['map_50']:.4f} map75: {final_map['map_75']:.4f} p: {final_map['precision']:.4f} r: {final_map['recall']:.4f} f1: {final_map['f1']:.4f}")

    print('training done')
    saveResultImages(model, val_loader, device, output_dir=f'runs/{run_name}/val_outputs/')
    save_metrics_plots(train_losses, val_losses, precisions, recalls, f1s, f'runs/{run_name}/plots/' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, help='run name')
    args = parser.parse_args()

    run_name = args.run

    if not os.path.exists(f'runs/{run_name}'):
        os.mkdir(f'runs/{run_name}')
        os.mkdir(f'runs/{run_name}/best_models')
        os.mkdir(f'runs/{run_name}/plots')
        os.mkdir(f'runs/{run_name}/val_outputs')

    train(run_name)

#TODO
"""
- inference.py
- save and load model
- print val_loss on each epoch 
"""





