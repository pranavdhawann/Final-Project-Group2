
import os
from ultralytics import YOLO

# === Configuration ===
# Update these paths before running
WEIGHTS = "runs/train/byu_training/weights/best.pt"
SOURCE = "/path/to/test_images"
OUTPUT_DIR = "runs/test_predict"
EXPERIMENT_NAME = "exp"

# Inference parameters
IMG_SIZE = 512
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
SAVE_TXT = True     # Save bounding boxes as YOLO-format .txt files
SAVE_CONF = True    # Include confidence scores in .txt labels

def main():
    # Create output directory
    save_path = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
    os.makedirs(save_path, exist_ok=True)

    # Load model
    model = YOLO(WEIGHTS)

    # Run inference
    _ = model.predict(
        source=SOURCE,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save_txt=SAVE_TXT,
        save_conf=SAVE_CONF,
        project=OUTPUT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True
    )

    print(f"Inference complete. Results saved to: {save_path}")

if __name__ == "__main__":
    main()
