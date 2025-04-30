import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from custom_dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

get_val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels']
))


def load_model(model_path, num_classes=2):
    model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        box_score_thresh=0.2,
        box_nms_thresh=0.3,
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=500,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean  # De-normalize


def generate_proper_saliency(model, device, img_tensor):
    img_tensor = img_tensor.to(device).requires_grad_()
    outputs = model([img_tensor])

    if len(outputs[0]['scores']) == 0:
        return None, None
    target_class = torch.argmax(outputs[0]['scores']).item()
    model.zero_grad()
    outputs[0]['scores'][target_class].backward()
    gradients = img_tensor.grad.data.abs()
    saliency, _ = gradients.max(dim=0)
    saliency = saliency.cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
    saliency = np.clip(saliency * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return saliency, heatmap


def create_overlay_visualization(original_img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlay


def process_images_with_proper_saliency(model, device, dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset))):
        try:
            img_tensor, _ = dataset[idx]
            img_display = denormalize_image(img_tensor)
            img_display = img_display.cpu().numpy().transpose(1, 2, 0)
            img_display = (img_display * 255).astype(np.uint8)
            saliency, heatmap = generate_proper_saliency(model, device, img_tensor)
            if saliency is None:
                continue
            overlay = create_overlay_visualization(
                cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR),
                heatmap
            )
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            ax1.imshow(img_display)
            ax1.set_title('Original Image')
            ax1.axis('off')
            ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax2.set_title('Saliency Heatmap Overlay')
            ax2.axis('off')

            plt.savefig(f"{output_dir}/saliency_{idx}.png", bbox_inches='tight', dpi=150)
            plt.close()

        except Exception as e:
            print(f"Skipping image {idx} due to error: {str(e)}")

if __name__ == "__main__":
    MODEL_PATH = "./runs/data-aug-1/best_models/best_model.pth"
    OUTPUT_DIR = "saliency_examples"
    model, device = load_model(MODEL_PATH)
    dataset = CustomDataset(split="val", transforms=get_val_transform)
    process_images_with_proper_saliency(model, device, dataset, OUTPUT_DIR)
    print(f"\nAll saliency maps saved to {OUTPUT_DIR}")