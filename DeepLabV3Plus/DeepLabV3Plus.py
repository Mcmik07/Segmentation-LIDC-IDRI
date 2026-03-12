import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module"""

    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.concat_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)
        x5 = self.conv5(x5)

        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.concat_conv(out)
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, input_channels=1):
        super(DeepLabV3Plus, self).__init__()

        # Utilisation de ResNet comme backbone
        resnet = models.resnet50(pretrained=True)

        # Modifier la première couche de ResNet pour accepter `input_channels` au lieu de 3
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # Modifier ici
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *list(resnet.children())[1:-2]
            # Utiliser uniquement l'encodeur de ResNet jusqu'à la couche avant le fully connected
        )

        # L'ASPP module
        self.aspp = ASPPModule(2048, 256)

        # Décodage (upsampling et convolution finale)
        self.decoder = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Passage dans le backbone ResNet
        x = self.backbone(x)

        # Passage dans l'ASPP module
        x = self.aspp(x)

        # Décodage (upsampling et convolution finale)
        out = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear',
                            align_corners=False)  # Upsample ici pour correspondre à la taille de la cible
        out = self.decoder(out)
        out = self.final_conv(out)

        # Upsample à la taille de la cible (512x512)
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)

        return out  # La sortie finale est maintenant "out"
