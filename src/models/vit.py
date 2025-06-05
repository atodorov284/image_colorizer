import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

from models.base_model import BaseColorizationModel


class ViTColorizationModel(BaseColorizationModel):
    def __init__(self, image_size: int = 224, pretrained: bool = True):
        """
        Constructor for the ViTColorizationModel class.

        Args:
            image_size (int): Size of the input image (assumed square).
                              The ViT B/16 is pre-trained on 224x224, but torchvision's
                              implementation can handle other sizes by interpolating positional embeddings.
            pretrained (bool): Whether to use pre-trained weights for the ViT encoder.
                               Defaults to True.
        """
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit_feature_extractor = models.vision_transformer.vit_b_16(
            weights=weights, image_size=image_size
        )

        self.patch_size = self.vit_feature_extractor.patch_size
        self.hidden_dim = self.vit_feature_extractor.hidden_dim
        self.num_patches_side = image_size // self.patch_size

        self.upsample_predict = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ViT colorization model.

        Args:
            x_l (torch.Tensor): Input tensor (e.g., L channel repeated 3 times for ViT).
                                Shape typically [Batch, 3, H, W].

        Returns:
            torch.Tensor: Predicted AB channels tensor. Shape [Batch, 2, H, W].
        """
        x = self.vit_feature_extractor._process_input(x_l)
        n = x.shape[0]
        batch_class_token = self.vit_feature_extractor.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit_feature_extractor.encoder(x)
        patch_embeddings = x[:, 1:, :]
        patch_embeddings = patch_embeddings.reshape(
            n, self.num_patches_side, self.num_patches_side, self.hidden_dim
        )
        features_2d = patch_embeddings.permute(
            0, 3, 1, 2
        ).contiguous()
        ab_output = self.upsample_predict(features_2d)
        return ab_output
