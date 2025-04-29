# Baseline Directory Overview

This document describes the contents of the baseline directory for our YOLOv8 project.

## 1. Scripts

1. **`yolov8-train-baseline.py`**  
   - Purpose: Train the YOLOv8 model on the motor‐detection dataset  
   - Usage example:  
     ```bash
     python yolov8-train-baseline.py 
     ```

2. **`yolov8-test-baseline.py`**  
   - Purpose: Run inference (testing) with a trained YOLOv8 model  
   - Usage example:  
     ```bash
     python yolov8-test-baseline.py 
     ```

## 2. Results Directory

- **`results/`**  
  - Contains all output artifacts from the baseline model:
    - `confusion_matrix_normalized.png`
    - `results.png` (loss & metric curves)
    - `R_curve.png` (Recall–Confidence)
    - `P_curve.png` (Precision–Confidence)
    - `PR_curve.png` (Precision–Recall)
    - `F1_curve.png` (F1–Confidence)


