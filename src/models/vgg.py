import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from models.base_model import BaseColorizationModel
from utils.colorization_utils import ColorizationUtils


class VGGColorizationModel(BaseColorizationModel):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )

        self.model5 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.Conv2d(
                512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True
            ),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )

        self.model8 = nn.Sequential(
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

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(
            ColorizationUtils.NUM_AB_BINS,
            2,
            kernel_size=1,
            padding=0,
            dilation=1,
            stride=1,
            bias=False,
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            vgg_features = vgg.features
            # model1: first conv (1->64), second conv (64->64)
            # model2: third conv (64->128), fourth conv (128->128)
            # Copy weights for model1[0] from vgg_features[0] (avg RGB)
            vgg_conv1_weight = vgg_features[0].weight.data  # (64, 3, 3, 3)
            vgg_conv1_bias = vgg_features[0].bias.data
            # Average across RGB channels
            avg_weight = vgg_conv1_weight.mean(dim=1, keepdim=True)  # (64, 1, 3, 3)
            self.model1[0].weight.data.copy_(avg_weight)
            self.model1[0].bias.data.copy_(vgg_conv1_bias)
            # model1[2] <- vgg_features[2], model2[0] <- vgg_features[5], model2[2] <- vgg_features[7]
            self.model1[2].weight.data.copy_(vgg_features[2].weight.data)
            self.model1[2].bias.data.copy_(vgg_features[2].bias.data)
            self.model2[0].weight.data.copy_(vgg_features[5].weight.data)
            self.model2[0].bias.data.copy_(vgg_features[5].bias.data)
            self.model2[2].weight.data.copy_(vgg_features[7].weight.data)
            self.model2[2].bias.data.copy_(vgg_features[7].bias.data)

    def forward(self, input_l):
        input_l = input_l[:, :1, :, :]
        conv1_2 = self.model1(ColorizationUtils.normalize_l_channel(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return ColorizationUtils.unnormalize_ab_channels(self.upsample4(out_reg))
