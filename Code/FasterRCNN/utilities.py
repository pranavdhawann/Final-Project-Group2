import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

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