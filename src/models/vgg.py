import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from models.base_model import BaseColorizationModel

NUM_AB_BINS = 313


class VGGColorizationModel(BaseColorizationModel):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None

        # Load the VGG16 model and select its feature extractor part
        vgg = vgg16(weights=weights)
        self.features = vgg.features

        # Decoder to upsample and predict color bins
        self.upsample_predict = nn.Sequential(
            # Input to the decoder will be the output of the VGG features (512 channels)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

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

            # Final convolution to get the 313 color bins
            nn.Conv2d(32, NUM_AB_BINS, kernel_size=3, padding=1),
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VGG-based colorization model.

        Args:
            x_l (torch.Tensor): Input LLL tensor. Shape: [Batch, 3, H, W].
                                The VGG model expects a 3-channel input.

        Returns:
            torch.Tensor: Predicted AB channel logits. Shape: [Batch, 313, H, W].
        """
        # Pass input through the VGG feature extractor
        features = self.features(x_l)

        # Pass features through the decoder to get color bin predictions
        ab_logits = self.upsample_predict(features)
        return ab_logits

