import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

from models.base_model import BaseColorizationModel

NUM_AB_BINS = 313


class ViTColorizationModel(BaseColorizationModel):
    def __init__(self, image_size: int = 224, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit_feature_extractor = models.vision_transformer.vit_b_16(
            weights=weights, image_size=image_size
        )

        self.patch_size = self.vit_feature_extractor.patch_size
        self.hidden_dim = self.vit_feature_extractor.hidden_dim
        self.num_patches_side = image_size // self.patch_size

        self.feature_extraction_layers = [2, 5, 8, 11]  # ViT-B/16 has 12 layers (0-11)
        num_in_channels = len(self.feature_extraction_layers) * self.hidden_dim

        # The paper's architecture (Fig 2) output ab distribution at H/4, W/4.
        # Assuming for now the upsampling here aims for full resolution to match resnet.
        self.upsample_predict = nn.Sequential(
            nn.Conv2d(num_in_channels, self.hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, NUM_AB_BINS, kernel_size=3, stride=1, padding=1),
            # Removed Tanh activation
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        x = self.vit_feature_extractor._process_input(x_l)
        n = x.shape[0]
        batch_class_token = self.vit_feature_extractor.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        intermediate_features = []
        for i, layer_module in enumerate(self.vit_feature_extractor.encoder.layers):
            x = layer_module(x)
            if i in self.feature_extraction_layers:
                patch_embeddings = x[:, 1:, :]
                patch_embeddings = patch_embeddings.reshape(
                    n, self.num_patches_side, self.num_patches_side, self.hidden_dim
                )
                features_2d = patch_embeddings.permute(0, 3, 1, 2).contiguous()
                intermediate_features.append(features_2d)

        fused_features = torch.cat(intermediate_features, dim=1)
        ab_logits = self.upsample_predict(fused_features)
        return ab_logits
