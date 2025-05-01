# Final-Project-Group2

**Course**: DATS 6303 – Deep Learning  
**Project**: Detection of Bacterial Flagellar Motors from Tomograms

## Overview

This project aims to detect bacterial flagellar motors in slices of tomograms using deep learning object detection models. We explore multiple architectures including YOLOv8, YOLOv10, CenterNet, and Faster R-CNN. The pipeline includes data preprocessing, annotation conversion, model training, evaluation, and visualization through a Streamlit dashboard.

## GitHub Directory Structure

```
.
├── .idea/                        # IDE settings
├── Code/                         # Source code for model training, preprocessing evaluation, and dashboard
├── Final-Group-Presentation/     # Group presentation slides
├── Final-Group-Project-Report/   # Final project report 
├── Group-Proposal/               # Initial project proposal
├── .gitignore                    # Git ignore file
├── README.md                     # Project overview 
├── requirements.txt              # Python dependencies
├── setup_dataset.sh              # Shell script to download/setup dataset
```

## Components

### 🔬 Models Used

- **YOLOv8**: Baseline for small object detection  
- **YOLOv10**: Improved detection with better spatial precision  
- **CenterNet**: Keypoint-based detection model  
- **Faster R-CNN**: Region-based object detection  

### 📊 Evaluation Metrics

- **mAP@50**  
- **mAP@50–95**  
- **Precision**  
- **Recall**

## Credits

- **Group Members**: Pranav Dhawan, Aakash Singh Sivaram, Abhinaysai Kamineni
- **University**: George Washington University  
- **Course Instructor**: Dr. Amir Jafari 
