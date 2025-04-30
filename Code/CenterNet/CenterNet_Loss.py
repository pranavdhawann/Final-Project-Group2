import torch
import torch.nn.functional as F

def focal_loss(preds, gt, alpha=2, beta=4):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    loss = 0
    pos_loss = torch.log(preds + 1e-6) * ((1 - preds) ** alpha) * pos_inds
    neg_loss = torch.log(1 - preds + 1e-6) * (preds ** alpha) * ((1 - gt) ** beta) * neg_inds

    num_pos = pos_inds.float().sum()
    loss = -(pos_loss + neg_loss).sum()
    return loss / (num_pos + 1e-6)

def l1_loss(preds, targets, mask):
    loss = F.l1_loss(preds * mask, targets * mask, reduction='sum')
    return loss / (mask.sum() + 1e-6)