import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
import random
from ultralytics import YOLO

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Configuration
TRUST = 4          # Number of slices around motor
BOX_SIZE = 24      # Bounding box size
TRAIN_SPLIT = 0.8  # Train/val split
DATA_PATH = "/home/ubuntu/yolo/BYU-dataset/"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
LABELS_CSV = os.path.join(DATA_PATH, "train_labels.csv")

# YOLO directory structure
YOLO_DATASET_DIR = "/home/ubuntu/yolo/yolo-dataset/"
os.makedirs(YOLO_DATASET_DIR, exist_ok=True)

# Create directories
yolo_images_train = os.path.join(YOLO_DATASET_DIR, "images", "train")
yolo_images_val = os.path.join(YOLO_DATASET_DIR, "images", "val")
yolo_labels_train = os.path.join(YOLO_DATASET_DIR, "labels", "train")
yolo_labels_val = os.path.join(YOLO_DATASET_DIR, "labels", "val")

for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

def normalize_slice(slice_data):
    """Normalize using 2nd and 98th percentiles."""
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped = np.clip(slice_data, p2, p98)
    return np.uint8(255 * (clipped - p2) / (p98 - p2))

def process_byu_dataset():
    """Process BYU dataset into YOLO format."""
    labels_df = pd.read_csv(LABELS_CSV)
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    np.random.shuffle(unique_tomos)
    split_idx = int(len(unique_tomos) * TRAIN_SPLIT)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    
    def process_tomo_group(tomos, img_dir, label_dir):
        count = 0
        for tomo_id in tqdm(tomos, desc="Processing BYU"):
            tomo_path = os.path.join(TRAIN_DIR, tomo_id)
            motors = tomo_df[tomo_df['tomo_id'] == tomo_id]
            
            for _, motor in motors.iterrows():
                z_center = int(motor['Motor axis 0'])
                z_min = max(0, z_center - TRUST)
                z_max = min(int(motor['Array shape (axis 0)']) - 1, z_center + TRUST)
                
                for z in range(z_min, z_max + 1):
                    slice_file = f"slice_{z:04d}.jpg"
                    src_path = os.path.join(tomo_path, slice_file)
                    if not os.path.exists(src_path):
                        continue
                        
                    img = Image.open(src_path)
                    normalized = normalize_slice(np.array(img))
                    dest_file = f"BYU_{tomo_id}_z{z:04d}.jpg"
                    Image.fromarray(normalized).save(os.path.join(img_dir, dest_file))
                    
                    # Convert coordinates to YOLO format
                    x_center = motor['Motor axis 2'] / img.width
                    y_center = motor['Motor axis 1'] / img.height
                    box_w = BOX_SIZE / img.width
                    box_h = BOX_SIZE / img.height
                    
                    label_file = dest_file.replace('.jpg', '.txt')
                    with open(os.path.join(label_dir, label_file), 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                    count += 1
        return count
    
    train_count = process_tomo_group(train_tomos, yolo_images_train, yolo_labels_train)
    val_count = process_tomo_group(val_tomos, yolo_images_val, yolo_labels_val)
    print(f"BYU dataset: {train_count} training, {val_count} validation slices")

def create_yaml_config():
    """Create YOLO dataset YAML configuration file."""
    yaml_content = {
        'path': YOLO_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    yaml_path = os.path.join(YOLO_DATASET_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    return yaml_path

# Training configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "img_size": 512,
    "pretrained_weights": "yolov8l.pt",
    "optimizer": "SGD",
    "lr0": 0.001,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "augment": True
}

def train_model(yaml_path):
    """Train YOLO model on BYU dataset."""
    model = YOLO(TRAINING_CONFIG["pretrained_weights"])
    
    results = model.train(
        data=yaml_path,
        epochs=TRAINING_CONFIG["epochs"],
        batch=TRAINING_CONFIG["batch_size"],
        imgsz=TRAINING_CONFIG["img_size"],
        optimizer=TRAINING_CONFIG["optimizer"],
        lr0=TRAINING_CONFIG["lr0"],
        momentum=TRAINING_CONFIG["momentum"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        augment=TRAINING_CONFIG["augment"],
        project="/home/ubuntu/yolo",
        name="byu_training",
        exist_ok=True
    )
    return model

# Main execution
if __name__ == "__main__":
    # Process dataset
    process_byu_dataset()
    
    # Create YAML config
    yaml_path = create_yaml_config()
    
    # Train model
    print("Starting training...")
    model = train_model(yaml_path)
    print("Training completed!")