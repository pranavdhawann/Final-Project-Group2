#!/usr/bin/env python3
import os
import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.ops import nms, generalized_box_iou
from tqdm import trange, tqdm

from CenterNet_Dataset import MotorDataset

SEED            = 420
EPOCHS          = 100
BATCH_SIZE      = 8
IMG_H, IMG_W    = 720, 720
OUTPUT_SIZE     = IMG_H // 4
DOWN_RATIO      = IMG_H // OUTPUT_SIZE
CSV_PATH        = "/home/ubuntu/Final-Project-Group2/Dataset/train_labels.csv"
IMG_DIR         = "/home/ubuntu/Final-Project-Group2/Dataset/train"
TRAIN_FRAC      = 0.8
VAL_FRAC        = 0.1
SAMPLE_SIZE     = 10000
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD  = 0.1
NMS_THRESHOLD   = 0.45
MAX_DETECTIONS  = 100
GIOU_LAMBDA     = 1.0
CLS_LAMBDA      = 0.5
PATIENCE        = 15

# Normalization tensors
MEAN = torch.tensor([0.485,0.456,0.406], device=DEVICE).view(1,3,1,1)
STD  = torch.tensor([0.229,0.224,0.225], device=DEVICE).view(1,3,1,1)

# Fix seeds
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if DEVICE.type=='cuda': torch.cuda.manual_seed_all(SEED)


class CenterNetFPN(nn.Module):
    def __init__(self):
        super().__init__()
        bb = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.conv1 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.l4 = nn.Conv2d(2048,256,1)
        self.l3 = nn.Conv2d(1024,256,1)
        self.l2 = nn.Conv2d(512,256,1)
        self.s3 = nn.Conv2d(256,256,3,padding=1)
        self.s2 = nn.Conv2d(256,256,3,padding=1)
        self.hm  = self._head(256,1)
        self.wh  = self._head(256,2)
        self.off = self._head(256,2)
        self.cls = self._head(256,1)
    def _head(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,out_c,1)
        )
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.l4(c5)
        p4 = self.s3(self.l3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest'))
        p3 = self.s2(self.l2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest'))
        feat = F.interpolate(p3, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear', align_corners=False)
        return (
            torch.sigmoid(self.hm(feat)),
            self.wh(feat),
            self.off(feat),
            self.cls(feat)
        )

def focal_loss_hnm(pred, tgt, alpha=2, beta=4):
    pred = pred.clamp(1e-6, 1-1e-6)
    pos = (tgt >= 0.5).float(); neg = (tgt < 0.5).float()
    num_pos = pos.sum().clamp(min=1)
    pos_loss = -(((1-pred)**alpha) * torch.log(pred) * pos).sum()
    neg_loss = -(((pred**alpha) * ((1-tgt)**beta)) * torch.log(1-pred) * neg).sum()
    return (pos_loss + neg_loss) / num_pos

def l1_loss(pred, tgt, mask):
    return (torch.abs(pred-tgt) * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-6)

def giou_loss(pb, gb):
    if pb.numel() == 0 or gb.numel() == 0:
        return torch.tensor(0., device=DEVICE)
    ious = generalized_box_iou(pb, gb)
    diag = ious.diag() if ious.shape[0] == ious.shape[1] else ious[:,0]
    return (1 - diag).mean()

def decode_topk(hm, wh, off, cls):
    B, _, H, W = hm.shape
    outs = []
    for b in range(B):
        heat = hm[b,0].view(-1)
        clsmap = cls[b,0].view(-1).sigmoid()
        score = heat * clsmap
        vals, ids = score.topk(MAX_DETECTIONS)
        boxes, scores, labels = [], [], []
        for v, i in zip(vals, ids):
            if v < CONF_THRESHOLD:
                break
            y, x = divmod(i.item(), W)
            w_, h_ = wh[b,0,y,x], wh[b,1,y,x]
            ox, oy = off[b,0,y,x], off[b,1,y,x]
            cx, cy = (x + ox) * DOWN_RATIO, (y + oy) * DOWN_RATIO
            x1, y1 = cx - w_/2, cy - h_/2
            x2, y2 = cx + w_/2, cy + h_/2
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(v)
            labels.append(1)
        if boxes:
            bt = torch.tensor(boxes, device=DEVICE)
            st = torch.tensor(scores, device=DEVICE)
            keep = nms(bt, st, NMS_THRESHOLD)
            bt, st = bt[keep], st[keep]
            labs = torch.tensor(labels, device=DEVICE)[keep]
        else:
            bt = torch.zeros((0,4), device=DEVICE)
            st = torch.zeros((0,), device=DEVICE)
            labs = torch.zeros((0,), dtype=torch.int64, device=DEVICE)
        outs.append({"boxes": bt, "scores": st, "labels": labs})
    return outs

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for imgs, ctr, det in loader:
        # multi-scale augment
        scale = random.uniform(0.8, 1.2)
        imgs = F.interpolate(imgs, scale_factor=scale, mode='bilinear', align_corners=False)
        imgs = F.interpolate(imgs, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
        imgs = imgs.to(DEVICE)
        hm_gt, sz_gt, off_gt, mask = [ctr[k].to(DEVICE) for k in ('heatmap','size','offset','mask')]
        optimizer.zero_grad()
        with autocast():
            hm_p, wh_p, off_p, cls_p = model((imgs - MEAN) / STD)
            l_hm  = focal_loss_hnm(hm_p, hm_gt)
            l_wh  = 0.1 * l1_loss(wh_p, sz_gt, mask)
            l_off = l1_loss(off_p, off_gt, mask)
            l_cls = F.binary_cross_entropy_with_logits(cls_p, hm_gt)
            preds = decode_topk(hm_p, wh_p, off_p, cls_p)
            l_gio = giou_loss(preds[0]['boxes'], det['boxes'][0].to(DEVICE))
            loss = l_hm + l_wh + l_off + CLS_LAMBDA*l_cls + GIOU_LAMBDA*l_gio
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_model(model, loader):
    model.eval()
    m50 = MeanAveragePrecision(iou_thresholds=[0.5]).to(DEVICE)
    m70 = MeanAveragePrecision(iou_thresholds=[0.7]).to(DEVICE)
    m90 = MeanAveragePrecision(iou_thresholds=[0.9]).to(DEVICE)
    y_t, y_p = [], []
    updates = 0
    with torch.no_grad():
        for imgs, ctr, det in loader:
            imgs = imgs.to(DEVICE)
            hm_p, wh_p, off_p, cls_p = model((imgs - MEAN) / STD)
            preds = decode_topk(hm_p, wh_p, off_p, cls_p)
            for b in range(len(det['boxes'])):
                gt = det['boxes'][b].to(DEVICE)
                pred_boxes = preds[b]['boxes']
                # guard empty cases
                if pred_boxes.numel() == 0 or gt.numel() == 0:
                    detected = False
                else:
                    iou_mat = generalized_box_iou(pred_boxes, gt)
                    detected = (iou_mat >= 0.5).any()
                y_t.append(1)
                y_p.append(int(detected))
            gts = []
            for b in range(len(det['boxes'])):
                gts.append({'boxes': det['boxes'][b].to(DEVICE), 'labels': det['labels'][b].to(DEVICE)})
            if any(p['boxes'].numel() > 0 for p in preds):
                m50.update(preds, gts)
                m70.update(preds, gts)
                m90.update(preds, gts)
                updates += 1
    map50 = m50.compute()['map'].item() if updates > 0 else 0.0
    map70 = m70.compute()['map'].item() if updates > 0 else 0.0
    map90 = m90.compute()['map'].item() if updates > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
    return map50, map70, map90, prec, rec, f1

if __name__ == '__main__':
    with open('training_log.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['ep','loss','mAP50','mAP70','mAP90','prec','rec','f1'])
    ds = MotorDataset(CSV_PATH, IMG_DIR, debug=False)
    idxs = torch.randperm(len(ds))[:SAMPLE_SIZE]
    t_end = int(TRAIN_FRAC * len(idxs))
    v_end = int((TRAIN_FRAC + VAL_FRAC) * len(idxs))
    train_dl = DataLoader(Subset(ds, idxs[:t_end]), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(Subset(ds, idxs[t_end:v_end]), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model    = CenterNetFPN().to(DEVICE)
    optimizer= optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler= CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler   = GradScaler()
    best_map = 0.0
    no_imp   = 0
    for ep in trange(1, EPOCHS+1, desc='Train'):
        t0      = time.time()
        loss    = train_epoch(model, train_dl, optimizer, scaler)
        map50, map70, map90, prec, rec, f1 = validate_model(model, val_dl)
        dt      = time.time() - t0
        tqdm.write(f"Ep{ep:03d} L:{loss:.4f} mAP50:{map50:.4f} mAP70:{map70:.4f} mAP90:{map90:.4f} "
                   f"P:{prec:.4f} R:{rec:.4f} F1:{f1:.4f} T:{dt:.1f}s")
        with open('training_log.csv','a',newline='') as f:
            csv.writer(f).writerow([ep, loss, map50, map70, map90, prec, rec, f1])
        if map50 > best_map:
            best_map = map50
            no_imp   = 0
            torch.save(model.state_dict(), 'centernet_best.pth')
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print('Early stopping')
                break
    print('Best mAP@0.5:', best_map)
