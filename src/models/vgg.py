import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from models.base_model import BaseColorizationModel
from utils.colorization_utils import ColorizationUtils


def make_conv_relu_bn_block(
    in_channels: int,
    out_channels: int,
    n: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    layers = nn.ModuleList()
    for i in range(n):
        layers.append(
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride if i == n - 1 else 1,
                padding=padding,
                dilation=dilation,
                bias=True,
            )
        )
        layers.append(nn.ReLU(True))
    layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class VGGColorizationModel(BaseColorizationModel):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.feature_blocks = nn.ModuleList(
            [
                make_conv_relu_bn_block(
                    in_channels=1,
                    out_channels=64,
                    n=2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                make_conv_relu_bn_block(
                    in_channels=64,
                    out_channels=128,
                    n=2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                make_conv_relu_bn_block(
                    in_channels=128,
                    out_channels=256,
                    n=3,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                make_conv_relu_bn_block(
                    in_channels=256,
                    out_channels=512,
                    n=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                make_conv_relu_bn_block(
                    in_channels=512,
                    out_channels=512,
                    n=3,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=2,
                ),
                make_conv_relu_bn_block(
                    in_channels=512,
                    out_channels=512,
                    n=3,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    dilation=2,
                ),
                make_conv_relu_bn_block(
                    in_channels=512,
                    out_channels=512,
                    n=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]
        )
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(
                256,
                ColorizationUtils.NUM_AB_BINS,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.regression_head = nn.Conv2d(
            ColorizationUtils.NUM_AB_BINS,
            2,
            kernel_size=1,
            padding=0,
            dilation=1,
            stride=1,
            bias=False,
        )
        self.upsample_layer = nn.Upsample(scale_factor=4, mode="bilinear")

        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            vgg_features = vgg.features
            vgg_conv_indices = [
                0,
                2,
                5,
                7,
                10,
                12,
                14,
                17,
                19,
                21,
                24,
                26,
                28,
            ]  # Conv2d layers in VGG16
            vgg_ptr = 0
            for block_idx, block in enumerate(self.feature_blocks):
                for layer_idx, layer in enumerate(block):
                    if isinstance(layer, nn.Conv2d):
                        if block_idx == 0 and layer_idx == 0:
                            # Special case: first conv, average RGB weights
                            vgg_conv1_weight = vgg_features[
                                vgg_conv_indices[vgg_ptr]
                            ].weight.data
                            vgg_conv1_bias = vgg_features[
                                vgg_conv_indices[vgg_ptr]
                            ].bias.data
                            avg_weight = vgg_conv1_weight.mean(dim=1, keepdim=True)
                            layer.weight.data.copy_(avg_weight)
                            layer.bias.data.copy_(vgg_conv1_bias)
                        else:
                            vgg_ptr += 1
                            if vgg_ptr >= len(vgg_conv_indices):
                                # Handle case where there are more feature blocks than VGG16 layers
                                break
                            layer.weight.data.copy_(
                                vgg_features[vgg_conv_indices[vgg_ptr]].weight.data
                            )
                            layer.bias.data.copy_(
                                vgg_features[vgg_conv_indices[vgg_ptr]].bias.data
                            )

    def forward(self, x_lll: torch.Tensor) -> torch.Tensor:
        x = x_lll[:, :1, :, :]
        x = ColorizationUtils.normalize_l_channel(x)
        for block in self.feature_blocks:
            x = block(x)
        out_reg = self.regression_head(nn.Softmax(dim=1)(self.transpose_conv(x)))
        return ColorizationUtils.unnormalize_ab_channels(self.upsample_layer(out_reg))
