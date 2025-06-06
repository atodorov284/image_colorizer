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

        # The upsampling path might need adjustment to ensure the final output resolution
        # matches the target H, W for the ab_logits if it's not already the case.
        # The paper's architecture (Fig 2) output ab distribution at H/4, W/4.
        # Assuming for now the upsampling here aims for full resolution to match resnet.
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
            nn.Conv2d(32, NUM_AB_BINS, kernel_size=3, stride=1, padding=1),
            # Removed Tanh activation
        )

    def forward(self, x_l: torch.Tensor) -> torch.Tensor:
        x = self.vit_feature_extractor._process_input(x_l)
        n = x.shape[0]
        batch_class_token = self.vit_feature_extractor.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit_feature_extractor.encoder(x)
        patch_embeddings = x[:, 1:, :]
        patch_embeddings = patch_embeddings.reshape(
            n, self.num_patches_side, self.num_patches_side, self.hidden_dim
        )
        features_2d = patch_embeddings.permute(0, 3, 1, 2).contiguous()
        # Output will be logits for the Q classes, shape (Batch, Q, H_feat, W_feat)
        # then upsampled by self.upsample_predict
        ab_logits = self.upsample_predict(features_2d)
        return ab_logits
