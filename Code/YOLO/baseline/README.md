# Baseline Directory Overview

This document describes the contents of the baseline directory for our YOLOv8 project.

## 1. Scripts

1. **`train-yolo.py`**  
   - Purpose: Train the YOLOv8 model on the motor‐detection dataset  
   - Usage example:  
     ```bash
     python train-yolo.py 
     ```

2. **`test-yolo.py`**  
   - Purpose: Run inference (testing) with a trained YOLOv8 model  
   - Usage example:  
     ```bash
     python test-yolo.py 
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


