import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from torchvision import transforms

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

def validate(model, val_loader, device):
    model.eval()
    iou_thresholds = [round(i * 0.05, 2) for i in range(10, 21)]
    metric_map = MeanAveragePrecision(iou_type="bbox", iou_thresholds=iou_thresholds)
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t  in targets]

            preds = model(images)
            metric_map.update(preds, targets)
    maps_ = metric_map.compute()
    return maps_


import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from torchvision import transforms


def saveResultImages(model, val_loader, device, output_dir='validation_results', transform=None,
                     images_per_composite=20):
    """
    Save visualization of predictions with 20 images per composite image (4 rows x 5 columns)

    Args:
        model: The detection model
        val_loader: Validation dataloader
        device: Device to run inference on
        output_dir: Directory to save results
        transform: Optional transform to apply to input images before visualization
        images_per_composite: Number of images per composite (20 for 4x5 grid)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Calculate grid dimensions (4 rows x 5 columns)
    rows = 4
    cols = 5

    composite_counter = 0
    image_counter = 0

    with torch.no_grad():
        # Initialize first composite figure
        fig, axs = plt.subplots(rows, cols, figsize=(25, 20))
        axs = axs.ravel()  # Flatten the axis array for easy indexing

        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            preds = model(images)

            for img_idx, (image, target, pred) in enumerate(zip(images, targets, preds)):
                # Apply transform if provided
                if transform:
                    image_to_show = transform(image.cpu())
                else:
                    # Default: convert tensor to numpy
                    image_to_show = image.cpu().permute(1, 2, 0).numpy()
                    if image_to_show.max() <= 1.0:  # if normalized
                        image_to_show = (image_to_show * 255).astype(np.uint8)

                # Get current axis in composite image
                ax = axs[image_counter % images_per_composite]
                ax.imshow(image_to_show)

                # Draw ground truth boxes (in green)
                for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                             edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, f'GT: {label}', color='white', fontsize=8,
                            bbox=dict(facecolor='green', alpha=0.7))

                # Draw prediction boxes (in red)
                for box, label, score in zip(pred['boxes'].cpu().numpy(),
                                             pred['labels'].cpu().numpy(),
                                             pred['scores'].cpu().numpy()):
                    if score < 0.5:  # only show confident predictions
                        continue
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1, f'P{label}:{score:.2f}', color='white', fontsize=8,
                            bbox=dict(facecolor='red', alpha=0.7))

                ax.axis('off')
                image_counter += 1

                # Save composite image when we have 20 images or at the end
                if image_counter % images_per_composite == 0 or (
                        batch_idx == len(val_loader) - 1 and img_idx == len(images) - 1):
                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'composite_{composite_counter}.jpg')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    composite_counter += 1

                    # Create new figure if there are more images to process
                    if image_counter < len(val_loader.dataset):
                        fig, axs = plt.subplots(rows, cols, figsize=(25, 20))
                        axs = axs.ravel()