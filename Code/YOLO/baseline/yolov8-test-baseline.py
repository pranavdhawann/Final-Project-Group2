import os
from ultralytics import YOLO

# === Configuration ===
# Update these paths before running
WEIGHTS = ""
SOURCE = ""
OUTPUT_DIR = "runs/test_predict"
EXPERIMENT_NAME = "exp_yolov8"

IMG_SIZE = 512
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
SAVE_TXT = True     
SAVE_CONF = True    

def main():
    save_path = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
    os.makedirs(save_path, exist_ok=True)

    model = YOLO(WEIGHTS)

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
