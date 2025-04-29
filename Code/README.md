---
title: "Code Folder"
---

This is the **Code** folder.

It contains all source code related to the final group project.


## PyCharm Setup
-  Click on `File` > `Project From Version Control`
-  Paste GitHub URL and create the project
-  SSH into your AWS account to the path `home\ubuntu`
-  (In AWS) `git clone https://github.com/pranavdhawann/Final-Project-Group2`
-  (In PyCharm) Click on `Tools` > `Deployment` > `Configurations` > `Mapping` > Click on the project folder in `home/ubuntu/Final-Project-Group2`

## Dataset Setup at AWS
- Go to your Kaggle account section and `Create New Token` this should download `kaggle.json` file to your system
- Move this into your local project folder amd rename it to `kaggle.json` and then upload to AWS deployment
- (IN AWS Terminal) `ls` and check if this file is uploaded into your AWS project folder as well
- (IN AWS Terminal) in project folder run `chmod +x setup_dataset.sh` then execute `./setup_dataset.sh`. `(Note: Join the kaggle competition before running)`
-  Previous execution should take a lot of time to download and unzip the dataset from kaggle and unzip it, grab a coffee meanwhile.

## PIP install issue in AWS
- If you are not able to download external pip packages run this on AWS terminal `sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED`

## Model Research:
### Phase 1 (Deadline Tuesday-18th Apr)
    - Train Faster RCNN and YOLO model.
    - With Hyper parameters
            - Epochs: 100
            - Batch Size: 4
            - Iput Image Size: (720x720x1)
            - Random seed: 420
            - Bounding box shape: (100px , 100px)
    - Deliverables
            - IoU
            - map@.5, map@.7, map@.9

## Results
### FasterRCNN:
#### 1. baseline:
    - Epochs: 20
    - Batch size: 4
    - map: 0.5284 map50: 0.8322 map75: 0.6948 p: 0.4150 r: 0.5676 f1: 0.4582
### 2. Image Ehnancement
```
def preprocessImg(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    denoised = cv2.medianBlur(gray, ksize=3)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    denoised = clahe.apply(denoised)
    denoised = cv2.fastNlMeansDenoising(
            denoised,
            h=15,
            templateWindowSize=7,
            searchWindowSize=21
    )
    return denoised
```
    - Epochs: 7 
    - Batch Size: 4
    - map: 0.5445 map50: 0.8309 map75: 0.7512 p: 0.5158 r: 0.5676 f1: 0.5315

### 3. Data Augmentation:
    - Epoch: 17
    - Batch Size: 10
    - map: 0.2674 map50: 0.7153 map75: 0.1505 p: 0.3420 r: 0.7368 f1: 0.4527