import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from .base_model import BaseColorizationModel  # Relative import


class ResNetColorizationModel(BaseColorizationModel):
    def __init__(self, pretrained: bool = True) -> None:
        """
        Constructor for the ResNetColorizationModel class.

        Args:
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        """
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.upsample_predict = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the colorization model.

        Args:
            x_l (torch.Tensor): Input tensor (e.g., L channel or LLL).
                                Shape typically [Batch, Channels, H, W].
        Returns:
            torch.Tensor: Predicted AB channels tensor. Shape [Batch, 2, H, W].
        """
        features = self.features(x_l)
        ab_output = self.upsample_predict(features)
        return ab_output
