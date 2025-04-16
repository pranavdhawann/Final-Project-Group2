import torch
import torch.nn as nn
import torchvision.models as models

class CenterNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(CenterNet, self).__init__()
        backbone = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 720 → 180

        self.hm_head = self._make_head(1)     # heatmap
        self.wh_head = self._make_head(2)     # width, height
        self.off_head = self._make_head(2)    # offset

    def _make_head(self, out_channels):
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "heatmap": torch.sigmoid(self.hm_head(feat)),
            "size": self.wh_head(feat),
            "offset": self.off_head(feat)
        }