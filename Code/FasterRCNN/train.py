import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
    collate_fn= lambda batch: tuple(zip(*batch)),
    num_workers= 4,
    pin_memory= True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
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


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0

    metric_map_50 = MeanAveragePrecision(iou_thresholds=[0.5])
    metric_map_60 = MeanAveragePrecision(iou_thresholds=[0.6])
    metric_map_70 = MeanAveragePrecision(iou_thresholds=[0.7])
    metric_map_90 = MeanAveragePrecision(iou_thresholds=[0.9])

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t  in targets]

            preds = model(images)
            metric_map_50.update(preds, targets)
            metric_map_60.update(preds, targets)
            metric_map_70.update(preds, targets)
            metric_map_90.update(preds, targets)

    map_50 = metric_map_50.compute()['map'].item()
    map_60 = metric_map_60.compute()['map'].item()
    map_70 = metric_map_70.compute()['map'].item()
    map_90 = metric_map_90.compute()['map'].item()

    print("=" * 50)
    print(f"mAP@50: {map_50:.4f}")
    print(f"mAP@60: {map_60:.4f}")
    print(f"mAP@70: {map_70:.4f}")
    print(f"mAP@90: {map_90:.4f}")
    print("=" * 50)

    # Optional: return them if you want to log/save
    return {
        'mAP@50': map_50,
        'mAP@60': map_60,
        'mAP@70': map_70,
        'mAP@90': map_90
    }




for epoch in range(num_epochs):

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

    lr_scheduler.step()

    print(f'Epoch: {epoch+1}, train_loss: {train_loss/len(train_loader):.4f}, val_loss: {val_loss/len(val_loader):.4f}')

    validate(model, val_loader, device)

print('training done')


#TODO
"""
- inference.py
- save and load model
- print val_loss on each epoch 
"""





