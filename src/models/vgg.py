import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

from models.base_model import BaseColorizationModel

NUM_AB_BINS = 313


class VGGColorizationModel(BaseColorizationModel):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0
        norm_layer = nn.BatchNorm2d

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
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
            norm_layer(512),
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
            norm_layer(512),
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(
            313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False
        )
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, input_l):
        input_l = input_l[:, :1, :, :]
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm
