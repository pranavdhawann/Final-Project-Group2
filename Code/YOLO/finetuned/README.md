# Baseline Directory Overview

This document describes the contents of the finetuned directory for our YOLOv10 model.

## 1. Scripts

1. **`yolov10-train-finetuned.py`**  
   - Purpose: Train the YOLOv10 model on the motor‐detection dataset  
   - Usage example:  
     ```bash
     python yolov10-train-finetuned.py 
     ```

2. **`yolov10-test.py`**  
   - Purpose: Run inference (testing) with a trained YOLOv10 model  
   - Usage example:  
     ```bash
     python yolov10-test.py 
     ```

## 2. Results Directory

- **`results/`**  
  - Contains all output artifacts from the finetuned model:
    - `confusion_matrix_normalized.png`
    - `R_curve.png` (Recall–Confidence)
    - `P_curve.png` (Precision–Confidence)
    - `PR_curve.png` (Precision–Recall)
    - `F1_curve.png` (F1–Confidence)
    - `dfl_loss_curve.png`


