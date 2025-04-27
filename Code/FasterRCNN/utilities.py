import pandas as pd
import torch
import matplotlib.patches as patches
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def get_frcnn_annotations(train_labels_path ,resize_shape):

    bb_w, bb_h = 100,100

    df = pd.read_csv(train_labels_path)
    grouped_list = [group for _, group in df.groupby(["tomo_id", "Motor axis 0"])]
    annotations = []     # {"filename", "boxes", labels}

    for group in grouped_list:
        file_name = group.iloc[0]["tomo_id"]+"/slice_"+str(int(group.iloc[0]["Motor axis 0"])).zfill(4)+".jpg" \
            if group.iloc[0]["Number of motors"] != 0 \
            else group.iloc[0]["tomo_id"]+"/slice_"+str(int(group.iloc[0]["Array shape (axis 0)"]/2)).zfill(4)+".jpg"
        boxes = []
        labels = []
        if group.iloc[0]["Number of motors"] != 0:
            for _, row in group.iterrows():
                cy = row["Motor axis 1"]
                cx = row["Motor axis 2"]

                x0,y0 = max(0, cx-bb_w/2), max(0, cy-bb_h/2)
                x1,y1 = min(row["Array shape (axis 2)"], cx+bb_w/2), min(row["Array shape (axis 1)"], cy+bb_h/2)

                x0,x1 = int((x0/row["Array shape (axis 2)"])*resize_shape[0]), int((x1/row["Array shape (axis 2)"])*resize_shape[0])
                y0,y1 = int((y0/row["Array shape (axis 1)"])*resize_shape[1]), int((y1/row["Array shape (axis 1)"])*resize_shape[1])

                boxes.append([x0,y0,x1,y1])
                labels.append(1)
        annotations.append(
            {
                "file_name": file_name,
                "boxes": boxes,
                "labels": labels,
                "split": group.iloc[0]["split"]
            }
        )
    return annotations

def save_image(img, target, save_path):
    fig, ax = plt.subplots(1)

    if isinstance(img, torch.Tensor):
        img_np = img.permute(1, 2, 0).cpu().numpy()
    else:
        raise TypeError("Image must be a PyTorch tensor")

    ax.imshow(img_np)

    for box in target["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Labels: {target['labels'].tolist()}")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def compute_metrics(detections, annotations, iou_threshold=0.5):

    detections = np.array(detections) if not isinstance(detections, np.ndarray) else detections
    annotations = np.array(annotations) if not isinstance(annotations, np.ndarray) else annotations
    true_positives = 0
    false_positives = 0
    num_annotations = len(annotations)
    matched_detections = [False] * len(detections)
    matched_annotations = [False] * len(annotations)
    for i, det in enumerate(detections):
        best_iou = 0
        best_ann_idx = -1

        for j, ann in enumerate(annotations):
            if matched_annotations[j]:
                continue

            iou = compute_iou(det, ann)
            if iou > best_iou:
                best_iou = iou
                best_ann_idx = j

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_detections[i] = True
            matched_annotations[best_ann_idx] = True
        else:
            false_positives += 1

    false_negatives = num_annotations - sum(matched_annotations)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / num_annotations if num_annotations > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def validate(model, val_loader, device):
    model.eval()
    iou_thresholds = [round(i * 0.05, 2) for i in range(10, 21)]
    metric_map = MeanAveragePrecision(iou_type="bbox", iou_thresholds=iou_thresholds)
    precisions, recalls, f1s = [], [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)
            metric_map.update(preds, targets)
            for idx, pred_ in enumerate(preds):
                det_ = preds[idx]["boxes"].cpu()
                gt_ = targets[idx]["boxes"].cpu()
                p,r,f1 = compute_metrics(det_, gt_, iou_threshold=0.5)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    maps_ = metric_map.compute()
    maps_["precision"] = avg_precision
    maps_["recall"] = avg_recall
    maps_["f1"] = avg_f1

    return maps_

def saveResultImages(model, val_loader, device, output_dir='validation_results', transform=None,
                     images_per_composite=20):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    rows = 4
    cols = 5

    composite_counter = 0
    image_counter = 0

    with torch.no_grad():
        fig, axs = plt.subplots(rows, cols, figsize=(25, 20))
        axs = axs.ravel()

        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)

            for img_idx, (image, target, pred) in enumerate(zip(images, targets, preds)):
                if transform:
                    image_to_show = transform(image.cpu())
                else:
                    image_to_show = image.cpu().permute(1, 2, 0).numpy()
                    if image_to_show.max() <= 1.0:  # if normalized
                        image_to_show = (image_to_show * 255).astype(np.uint8)
                ax = axs[image_counter % images_per_composite]
                ax.imshow(image_to_show)
                for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                             edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, f'GT: {label}', color='white', fontsize=8,
                            bbox=dict(facecolor='green', alpha=0.7))
                for box, label, score in zip(pred['boxes'].cpu().numpy(),
                                             pred['labels'].cpu().numpy(),
                                             pred['scores'].cpu().numpy()):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, f'P{label}:{score:.2f}', color='white', fontsize=8,
                            bbox=dict(facecolor='red', alpha=0.7))

                ax.axis('off')
                image_counter += 1
                if image_counter % images_per_composite == 0 or (
                        batch_idx == len(val_loader) - 1 and img_idx == len(images) - 1):
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'composite_{composite_counter}.jpg')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    composite_counter += 1
                    if image_counter < len(val_loader.dataset):
                        fig, axs = plt.subplots(rows, cols, figsize=(25, 20))
                        axs = axs.ravel()



def save_metrics_plots(train_losses, val_losses, precisions, recalls, f1s, output_dir="plots"):
    """
    Save loss and metric plots to a directory

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        precisions: List of precision values
        recalls: List of recall values
        f1s: List of F1 scores
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set modern style (works across matplotlib versions)
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 12
    })

    # 1. Plot Losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot Metrics
    plt.figure(figsize=(10, 6))
    plt.plot(precisions, label='Precision', linewidth=2, color='blue')
    plt.plot(recalls, label='Recall', linewidth=2, color='green')
    plt.plot(f1s, label='F1 Score', linewidth=2, color='red')
    plt.title('Detection Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()


import cv2
def preprocessImg(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    denoised = cv2.medianBlur(gray, ksize=3)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    denoised = clahe.apply(denoised)
    denoised = cv2.fastNlMeansDenoising(
            denoised,
            h=15,
            templateWindowSize=7,
            searchWindowSize=21
    )
    return denoised