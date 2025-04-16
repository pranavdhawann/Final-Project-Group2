from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.7, 0.9])
metric.update(preds=[{
    "boxes": pred_boxes,  # Tensor[N, 4]
    "scores": conf_scores,
    "labels": labels
}], target=[{
    "boxes": gt_boxes,
    "labels": labels
}])

results = metric.compute()
print("mAP@0.5:", results["map_50"])
print("mAP@0.7:", results["map_70"])
print("mAP@0.9:", results["map_90"])