import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
from PIL import Image
from ultralytics import YOLO
import streamlit as st
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ---- CONSTANTS ----
MAX_DETECTIONS_PER_TOMO = 3
NMS_IOU_THRESHOLD = 0.2
MODEL_PATHS = {
    "YOLOv10": "models/YOLO/best.pt",  
    "CenterNet": "models/CenterNet/centernet_final.pth",
    "Faster R-CNN": "models/FasterRCNN/best_model.pth"
}

warnings.filterwarnings("ignore")

# Added CenterNet specific constants
IMG_H, IMG_W = 720, 720
BOX_W, BOX_H = 100, 100
OUTPUT_SIZE = IMG_H // 4  
DOWN_RATIO = IMG_H // OUTPUT_SIZE

# ---- UTILS ----
def normalize_slice(slice_data):
    p2, p98 = np.percentile(slice_data, [2, 98])
    clipped = np.clip(slice_data, p2, p98)
    return np.uint8(255 * (clipped - p2) / (p98 - p2 + 1e-7))

def preload_image_batch(file_paths):
    return [cv2.imread(p) if cv2.imread(p) is not None else np.array(Image.open(p)) for p in file_paths]

def create_tensor(img_array, device):
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    return tensor.to(device).unsqueeze(0)

# ---- CENTERNET PREPROCESSING ----
def gaussian2D(shape, sigma=1):
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hm, center, r):
    """Draw gaussian peak on heatmap at specified center with radius r."""
    d = 2 * r + 1
    g = gaussian2D((d, d), sigma=d / 6)

    x, y = int(center[0]), int(center[1])

    if x < -d // 2 or y < -d // 2 or x >= hm.shape[1] + d // 2 or y >= hm.shape[0] + d // 2:
        return

    left, right = max(0, x - r), min(hm.shape[1], x + r + 1)
    top, bottom = max(0, y - r), min(hm.shape[0], y + r + 1)

    if left >= right or top >= bottom:
        return

    g_left = max(0, -x + r)
    g_right = g_left + (right - left)
    g_top = max(0, -y + r)
    g_bottom = g_top + (bottom - top)

    gaussian_patch = g[g_top:g_bottom, g_left:g_right]
    hm_patch = hm[top:bottom, left:right]

    if gaussian_patch.shape[0] > 0 and gaussian_patch.shape[1] > 0:
        if gaussian_patch.shape == hm_patch.shape:
            np.maximum(hm_patch, gaussian_patch, out=hm_patch)
        else:
            print(f"Shape mismatch: hm_patch {hm_patch.shape}, g_patch {gaussian_patch.shape}")


def create_centernet_targets(boxes, debug=False):
    hm = np.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    size = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    off = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), np.float32)

    valid_boxes = 0

    for box_idx, (x, y, w, h) in enumerate(boxes):
        if x + w <= 0 or y + h <= 0 or x >= IMG_W or y >= IMG_H:
            if debug:
                print(f"Skipping box {box_idx} completely outside image: {[x, y, w, h]}")
            continue

        x_orig, y_orig = x, y
        x = max(0, min(x, IMG_W - 1))
        y = max(0, min(y, IMG_H - 1))
        w = max(1, min(w, IMG_W - x))
        h = max(1, min(h, IMG_H - y))

        if debug and (x != x_orig or y != y_orig):
            print(f"Adjusted box {box_idx} from {[x_orig, y_orig, w, h]} to {[x, y, w, h]}")

        cx, cy = x + w / 2, y + h / 2
        cx_out, cy_out = cx / DOWN_RATIO, cy / DOWN_RATIO

        if cx_out < 0 or cx_out >= OUTPUT_SIZE or cy_out < 0 or cy_out >= OUTPUT_SIZE:
            if debug:
                print(f"Box {box_idx} center ({cx_out:.1f}, {cy_out:.1f}) outside output bounds")
            continue

        xi, yi = int(cx_out), int(cy_out)
        r = max(2, int(min(w, h) / (DOWN_RATIO * 2)))

        if debug:
            print(f"Drawing gaussian at ({cx_out:.1f}, {cy_out:.1f}) with radius {r}")
        draw_gaussian(hm[0], (cx_out, cy_out), r)

        size[:, yi, xi] = [w, h]
        off[:, yi, xi] = [cx_out - xi, cy_out - yi]
        mask[yi, xi] = 1
        valid_boxes += 1

    if debug:
        print(f"Created targets with {valid_boxes}/{len(boxes)} valid boxes")
        print(f"Heatmap stats - min: {hm.min():.4f}, max: {hm.max():.4f}, mean: {hm.mean():.4f}")
        print(f"Valid locations: {mask.sum()}")

    return hm, size, off, mask

def preprocess_for_centernet(img, device):
    """Preprocess image for CenterNet model input."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
    return img_tensor.to(device).unsqueeze(0)

def decode_centernet_output(hm, size, offset, conf_threshold):
    """Decode CenterNet output heatmap, size, and offset into bounding boxes."""
    # Extract single-channel heatmap if needed
    if hm.ndim == 3:
        hm = hm[0]
    detections = []
    ys, xs = np.where(hm >= conf_threshold)
    for y, x in zip(ys, xs):
        conf = float(hm[y, x])
        off_x = offset[0, y, x]
        off_y = offset[1, y, x]
        cx = (x + off_x) * DOWN_RATIO
        cy = (y + off_y) * DOWN_RATIO
        w = size[0, y, x]
        h = size[1, y, x]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        detections.append([x1, y1, x2, y2, conf])
    return detections

def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "YOLOv10":
        model = YOLO(path).to(device)
        model.fuse()
        return model.eval(), device

    if model_name == "CenterNet":
        from CenterNet.CenterNet_Model import CenterNet
        model = CenterNet(num_classes=1)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return model.to(device).eval(), device

    # Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), device

def annotate(image, detections, color=(0,255,0), thickness=2):
    out = image.copy()
    for det in detections:
        x1, y1 = int(det['xmin']), int(det['ymin'])
        x2, y2 = int(det['xmax']), int(det['ymax'])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out

def display_columns(orig, proc, orig_cap, proc_cap):
    c1, c2 = st.columns(2)
    c1.image(orig, caption=orig_cap, use_container_width=True)
    c2.image(proc, caption=proc_cap, use_container_width=True)


def create_tensor(img_bgr, device=None):
    """Convert BGR image to normalized tensor"""
    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW

    # Normalize with ImageNet stats
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)

    # Move to device if specified
    if device:
        img_tensor = img_tensor.to(device)
    return img_tensor.unsqueeze(0)  # Add batch dimension

# ---- MAIN APP ----
def main():
    st.markdown("""
        <h1 style='text-align:center; color:#2E86C1;'>🦠 Bacterial Flagellar Motor Detection</h1>
        <p style='text-align:center; color:#5D6D7E;'>Automated detection on your tomogram images</p>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Select Model", list(MODEL_PATHS))
        confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.0, 0.01)

    model, device = load_model(model_name)

    # YOLOv10 branch
    if model_name == "YOLOv10":
        files = st.file_uploader(
            "Drag & Drop Tomogram Image(s)",
            type=["jpg", "jpeg", "png", "tiff"],
            accept_multiple_files=True
        )
        if not files:
            st.stop()

        st.subheader("YOLOv10 Detection Results")
        for f in files:
            img = Image.open(f).convert('RGB')
            img_resized = img.resize((960, 960))
            img_np = np.array(img_resized)

            results = model(img_np,
                            conf=confidence,
                            iou=NMS_IOU_THRESHOLD,
                            imgsz=960,
                            verbose=False)

            detections = []
            if results:
                result = results[0]
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    if box.conf >= confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2,
                            "confidence": float(box.conf)
                        })

            proc_img = annotate(img_np, detections)
            display_columns(img_resized, proc_img, f"Original: {f.name}", "Detected")

            with st.expander("Detection Details"):
                st.write(f"Total detections: {len(detections)}")
                if detections:
                    df = pd.DataFrame(detections)
                    st.dataframe(df.style.highlight_max(axis=0, color='#d8f3dc'))

    # Faster R-CNN branch
    elif model_name == "Faster R-CNN":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        import torchvision.transforms as T
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # (Re)initialize and load weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 2  # change as needed
        model_frcnn = fasterrcnn_resnet50_fpn(
            pretrained=False,
            box_score_thresh=0.2,
            box_nms_thresh=0.3,
            rpn_pre_nms_top_n_train=1000,
            rpn_post_nms_top_n_train=500,
        )
        in_features = model_frcnn.roi_heads.box_predictor.cls_score.in_features
        model_frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model_frcnn.load_state_dict(torch.load('./models/FasterRCNN/best_model.pth',
                                                map_location=device))
        model_frcnn.to(device).eval()

        files = st.file_uploader(
            "Drag & Drop Tomogram Image(s)",
            type=["jpg", "jpeg", "png", "tiff"],
            accept_multiple_files=True
        )
        if not files:
            st.stop()

        st.subheader("Faster R-CNN Detection Results")
        transform = T.Compose([T.ToTensor()])

        for f in files:
            img = Image.open(f).convert('RGB')
            img_resized = img.resize((900, 900))
            img_np = np.array(img_resized)
            img_tensor = transform(img_np).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model_frcnn(img_tensor)[0]

            detections = []
            for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
                if score >= confidence:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    detections.append({
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "confidence": float(score),
                        "class": int(label)
                    })

            proc_img = annotate(img_np, detections)
            display_columns(img_resized, proc_img, f"Original: {f.name}", "Detected")

            with st.expander("Detection Details"):
                st.write(f"Total detections: {len(detections)}")
                if detections:
                    df = pd.DataFrame(detections)
                    st.dataframe(df.style.highlight_max(axis=0, color='#d8f3dc'))

    # CenterNet branch
    elif model_name == "CenterNet":
        f = st.file_uploader(
            "Drag & Drop Tomogram Image",
            type=["jpg", "jpeg", "png", "tiff"]
        )
        if not f:
            st.stop()

        img_gray = Image.open(f).convert('L')
        arr = np.array(img_gray)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        with st.spinner('Running CenterNet detection...'):
            tensor = preprocess_for_centernet(arr, device)
            out = model(tensor)
            raw = decode_centernet_output(
                out['heatmap'][0].detach().cpu().numpy(),
                out['size'][0].detach().cpu().numpy(),
                out['offset'][0].detach().cpu().numpy(),
                confidence
            )
            detections = [
                {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
                for x1, y1, x2, y2, conf in raw
            ]
        proc_img = annotate(img_bgr, detections)
        display_columns(img_gray, proc_img, "Original", "Processed")

        st.subheader("📊 Detection Statistics")
        if detections:
            df = pd.DataFrame(detections)
            st.write(df, unsafe_allow_html=True)
        else:
            st.warning("No detections")

    # Generic fallback
    else:
        f = st.file_uploader(
            "Drag & Drop Tomogram Image",
            type=["jpg", "jpeg", "png", "tiff"]
        )
        if not f:
            st.stop()

        img_gray = Image.open(f).convert('L')
        arr = np.array(img_gray)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        with st.spinner(f'Running {model_name} detection...'):
            tensor = create_tensor(img_bgr, model.device)
            outs = model(tensor)[0]
            detections = []
            for box, score in zip(outs['boxes'], outs['scores']):
                if score >= confidence:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    detections.append({
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2
                    })

            proc_img = annotate(img_bgr, detections)
            display_columns(img_gray, proc_img, "Original", "Processed")

        st.subheader("📊 Detection Statistics")
        if detections:
            df = pd.DataFrame(detections)
            st.write(df, unsafe_allow_html=True)
        else:
            st.warning("No detections")


if __name__ == "__main__":
    main()
