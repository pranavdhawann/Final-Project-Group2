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
- (IN AWS Terminal) in project folder run `chmod +x setup_dataset.sh` then execute `./setup_dataset.sh`
-  Previous execution should take a lot of time to download and unzip the dataset from kaggle and unzip it, grab a coffee meanwhile.

## PIP install issue in AWS
- If you are not able to download external pip packages run this on AWS terminal `sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED`