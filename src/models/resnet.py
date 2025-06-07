import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34

from models.base_model import BaseColorizationModel

NUM_AB_BINS = 313


class ResNetColorizationModel(BaseColorizationModel):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.upsample_predict = nn.Sequential(
            # The output of the full ResNet-18/34 is 512 channels
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, NUM_AB_BINS, kernel_size=3, padding=1),
            # Removed Tanh activation
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        features = self.features(x_l)
        # Output will be logits for the Q classes, shape (Batch, Q, H, W)
        ab_logits = self.upsample_predict(features)
        return ab_logits