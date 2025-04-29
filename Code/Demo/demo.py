import os
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

# ---- CONSTANTS ----
MAX_DETECTIONS_PER_TOMO = 3
NMS_IOU_THRESHOLD = 0.2
MODEL_PATHS = {
    "YOLOv10": "models/YOLO/best.pt",  
    "CenterNet": "models/CenterNet/centernet_final.pth",
    "Faster R-CNN": "models/FasterRCNN/best_model.pth"
}

warnings.filterwarnings("ignore")

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

def load_model(model_name):
    path = MODEL_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "YOLOv10": 
        model = YOLO(path).to(device)
        model.fuse()
        return model.eval()

    if model_name == "CenterNet":
        from CenterNet.CenterNet_Model import CenterNet, decode_centernet_output
        model = CenterNet(num_classes=1)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return (model.to(device).eval(), decode_centernet_output)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

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

    model_info = load_model(model_name)
    if model_name == "CenterNet":
        model, decode_centernet_output = model_info
    else:
        model = model_info

    if model_name == "YOLOv10":  
        files = st.file_uploader("Drag & Drop Tomogram Image(s)",
                               type=["jpg","jpeg","png","tiff"],
                               accept_multiple_files=True)
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
    elif model_name == "Faster R-CNN":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        import torchvision.transforms as T
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        # Load the model first
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model architecture
        model = fasterrcnn_resnet50_fpn(
            pretrained=False,  # We're loading our own weights
            box_score_thresh=0.2,
            box_nms_thresh=0.3,
            rpn_pre_nms_top_n_train=1000,
            rpn_post_nms_top_n_train=500,
        )

        # Modify for your custom classes
        num_classes = 2  # Update this with your actual number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load the trained weights
        model.load_state_dict(torch.load('./models/FasterRCNN/best_model.pth', map_location=device))
        model.to(device)
        model.eval()

        # File uploader
        files = st.file_uploader("Drag & Drop Tomogram Image(s)",
                                 type=["jpg", "jpeg", "png", "tiff"],
                                 accept_multiple_files=True)
        if not files:
            st.stop()

        st.subheader("Faster R-CNN Detection Results")

        # Define transforms
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        for f in files:
            # Load and preprocess image
            img = Image.open(f).convert('RGB')
            img_resized = img.resize((900, 900))
            img_np = np.array(img_resized)

            # Apply transforms
            img_tensor = transform(img_np).unsqueeze(0).to(device)
            print(img_tensor.shape)
            # Run inference
            with torch.no_grad():
                print(model(img_tensor))
                predictions = model(img_tensor)[0]

            # Process detections
            detections = []
            for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
                if score >= confidence:  # Apply confidence threshold
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    detections.append({
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "confidence": float(score),
                        "class": label.item()
                    })

            # Visualize results
            proc_img = annotate(img_np, detections)
            display_columns(img_resized, proc_img, f"Original: {f.name}", "Detected")

            with st.expander("Detection Details"):
                st.write(f"Total detections: {len(detections)}")
                if detections:
                    df = pd.DataFrame(detections)
                    st.dataframe(df.style.highlight_max(axis=0, color='#d8f3dc'))
    else:  
        f = st.file_uploader("Drag & Drop Tomogram Image", type=["jpg","jpeg","png","tiff"])
        if not f:
            st.stop()

        img_gray = Image.open(f).convert('L')
        arr = np.array(img_gray)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        with st.spinner(f'Running {model_name} detection...'):
            tensor = create_tensor(img_bgr, model.device)
            if model_name == "CenterNet":
                out = model(tensor)
                raw = decode_centernet_output(
                    out['heatmap'][0].cpu().numpy(),
                    out['size'][0].cpu().numpy(),
                    out['offset'][0].cpu().numpy(),
                    confidence
                )
                detections = [{'xmin':x1,'ymin':y1,'xmax':x2,'ymax':y2}
                            for x1,y1,x2,y2,conf in raw]
            else:
                outs = model(tensor)[0]
                detections = []
                for box, score in zip(outs['boxes'], outs['scores']):
                    if score >= confidence:
                        x1,y1,x2,y2 = box.cpu().numpy().astype(int)
                        detections.append({'xmin': x1,'ymin': y1,'xmax': x2,'ymax': y2})

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