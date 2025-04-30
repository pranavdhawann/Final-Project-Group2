# Bacterial Flagellar Motor Detection with CenterNet

## 1. Approach & Pipeline

1. **Data Preparation**  
   - Parsed tomogram metadata CSV to extract motor center coordinates.  
   - Generated fixed-size bounding boxes (100×100) around each motor axis.  
   - Loaded corresponding 2D slices as grayscale images and resized to 720×720.

2. **Data Augmentation & Preprocessing**  
   - Applied random zoom, translation, rotation, flips, brightness/contrast jitter, CLAHE, and Gaussian noise to both images and boxes.  
   - Normalized images to zero-mean, unit-variance and clipped extreme values to ±3σ.  
   - Generated CenterNet targets: heatmap, size, offset, and mask maps at 180×180 resolution.

3. **Model & Training**  
   - **Backbone:** ResNet-18 pretrained on ImageNet; modified detection heads for heatmap, width-height, and offset prediction.  
   - **Losses:** Focal loss for heatmap; L1 loss for size/offset.  
   - **Optimization:** Adam optimizer with cosine annealing LR schedule; mixed‐precision training.

4. **Utils & Other Evaluation**  
   - Decoded top‐K heatmap peaks, applied NMS, and filtered by size constraints.  
   - Evaluated using IoU‐based mAP at thresholds 0.5, 0.7, 0.9, plus precision, recall, and F1 (β=1).

## 2. Key Results (Final Evaluation)

| Metric           | Value   |
|------------------|:-------:|
| **mAP@50**       | 49.2%   |
| **mAP@70**       | 35.6%   |
| **mAP@90**       | 1.4%    |
| **Precision**    | 73.5%   |
| **Recall**       | 42.8%   |
| **F1 Score**     | 54.2%   |

{Currently Way Forwards are implemented so the metrics value is Subject to change.}

## 3. Way Forward

1. **Multi‐Scale Feature Fusion**  
   Incorporate feature pyramids (FPN) or dilated convolutions to better detect motors at varying scales and improve mAP@70/90.

2. **Stronger Backbone & Losses**  
   Upgrade to ResNet-50/101 or use transformer‐based backbones (e.g. Swin Transformer) and integrate IoU‐guided losses for tighter localization.

3. **Advanced Augmentation & Semi‐Supervision**  
   Leverage synthetic tomogram generation, context‐aware perturbations, and pseudo‐labeling on unlabeled data to enrich training diversity and recall robustness.
