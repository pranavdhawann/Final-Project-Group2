# Demo

This repository contains model files, test images, and an inference demo for detecting bacterial flagellar motors in tomograms.

This folder contains the Streamlit demo for the project. The main demo script is **demo.py**, which launches the interactive app.

## Directory Structure

. 
├── models/ # Trained model weights (YOLO, CenterNet, Faster R-CNN) 
├── test-images/ # Sample input tomogram slices for inference 
├── demo.py # Streamlit app to run inference 
├── README.md 

## Steps to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2. Navigate to the `demo` directory:  
   ```bash
   cd ..../demo
   ```
3. Run the demo with Streamlit:  
   ```bash
   streamlit run demo.py
   ```
