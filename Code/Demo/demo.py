import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
import cv2
from ultralytics import YOLO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        color: #2E86C1;
        text-align: center;
        padding: 20px;
    }
    .subheader {
        font-size: 20px !important;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #5D6D7E;
        border-radius: 5px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .stSlider > div > div > div {
        background: #2E86C1 !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATHS = {
    "YOLOv8": "models/YOLO/best.pt",
    "CenterNet": "models/CenterNet/centernet_final.pth",
    "Faster R-CNN": "models/FasterRCNN/best_model.pth"
}

@st.cache_resource
def load_model(model_name):
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if model_name == "YOLOv8":
            model = YOLO(model_path)
            model.to(device)
        elif model_name == "CenterNet":
            from CenterNet.CenterNet_Model import CenterNet
            model = CenterNet(num_classes=1)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        else:  # Faster R-CNN
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def decode_centernet_output(heatmap, size, offset, confidence_threshold, down_ratio=4):
    """Decode CenterNet outputs to bounding boxes"""
    # Get peak locations
    heatmap = heatmap.squeeze()
    peaks = np.where(heatmap >= confidence_threshold)
    
    detections = []
    for y, x in zip(peaks[0], peaks[1]):
        # Get raw predictions
        dx = offset[0, y, x].item()
        dy = offset[1, y, x].item()
        w = size[0, y, x].item()
        h = size[1, y, x].item()
        
        # Convert to original image coordinates
        x_center = (x + dx) * down_ratio
        y_center = (y + dy) * down_ratio
        width = w * down_ratio
        height = h * down_ratio
        
        detections.append({
            'xmin': x_center - width/2,
            'ymin': y_center - height/2,
            'xmax': x_center + width/2,
            'ymax': y_center + height/2,
            'confidence': heatmap[y, x],
            'class': 0,
            'name': 'flagellar_motor'
        })
    
    return pd.DataFrame(detections)

def main():
    st.markdown('<p class="header">🦠 Bacterial Flagellar Motor Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Automated detection of flagellar motors in cryo-electron tomograms</p>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Select Detection Model",
            tuple(MODEL_PATHS.keys())
        )
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )

    # File upload section
    st.markdown('<p class="subheader">Drag and drop your tomogram image below</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        " ",
        type=["jpg", "jpeg", "png", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Load as grayscale
        image = Image.open(uploaded_file).convert('L')
        img_array = np.array(image)

        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Tomogram", use_container_width=True, clamp=True)

        # Convert grayscale to 3-channel for models
        yolo_input = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        other_input = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Detection processing
        with st.spinner(f'🔍 Running {model_name} detection...'):
            try:
                model = load_model(model_name)
                device = next(model.parameters()).device  # Get device from model
                
                if model_name == "YOLOv8":
                    results = model(yolo_input, conf=confidence_threshold)
                    detections = results[0].boxes.data.cpu().numpy()
                    columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
                    detections = pd.DataFrame(detections, columns=columns)
                    detections['class'] = detections['class'].astype(int)
                    detections['name'] = detections['class'].apply(lambda x: model.names[x])
                    detected_img = results[0].plot()[:, :, ::-1]
                elif model_name == "CenterNet":
                    # Prepare input tensor
                    input_tensor = torch.from_numpy(other_input).permute(2, 0, 1).float()
                    input_tensor = input_tensor.to(device).unsqueeze(0) / 255.0
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(input_tensor)
                    
                    # Convert outputs to numpy
                    heatmap = outputs['heatmap'].cpu().numpy()[0]
                    size = outputs['size'].cpu().numpy()[0]
                    offset = outputs['offset'].cpu().numpy()[0]
                    
                    # Decode predictions
                    detections = decode_centernet_output(heatmap, size, offset, confidence_threshold)
                    
                    # Draw bounding boxes
                    detected_img = other_input.copy()
                    for _, det in detections.iterrows():
                        x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                        cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['name']} {det['confidence']:.2f}"
                        cv2.putText(detected_img, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:  # Faster R-CNN
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(other_input).permute(2, 0, 1).float()
                        input_tensor = input_tensor.to(device).unsqueeze(0) / 255.0
                        outputs = model(input_tensor)[0]
                    
                    detections = []
                    for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
                        if score >= confidence_threshold:
                            detections.append({
                                'xmin': box[0].item(),
                                'ymin': box[1].item(),
                                'xmax': box[2].item(),
                                'ymax': box[3].item(),
                                'confidence': score.item(),
                                'class': label.item(),
                                'name': 'flagellar_motor'
                            })
                    detections = pd.DataFrame(detections)
                    
                    # Draw bounding boxes
                    detected_img = other_input.copy()
                    for _, det in detections.iterrows():
                        x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                        cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['name']} {det['confidence']:.2f}"
                        cv2.putText(detected_img, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                with col2:
                    st.image(detected_img, caption="Detection Results", use_container_width=True)

                st.subheader("📊 Detection Statistics")
                st.write(f"Total detections: {len(detections)}")
                if not detections.empty:
                    st.dataframe(detections.style.highlight_max(axis=0, color='#EBF5FB'))
                else:
                    st.warning("No detections meeting confidence threshold")

            except Exception as e:
                st.error(f"Error during detection: {e}")

if __name__ == "__main__":
    main()