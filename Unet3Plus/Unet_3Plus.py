import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class UNet3Plus(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, feature_scale=1):
        super().__init__()
        filters = [32, 64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]

        # Encoder
        self.conv0 = VGGBlock(input_channels, filters[0], filters[0])
        self.pool0 = nn.MaxPool2d(2)
        self.conv1 = VGGBlock(filters[0], filters[1], filters[1])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = VGGBlock(filters[1], filters[2], filters[2])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = VGGBlock(filters[2], filters[3], filters[3])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = VGGBlock(filters[3], filters[4], filters[4])

        # Decoder
        cat_channels = filters[0]
        up_channels = filters[0]

        self.conv_blocks = nn.ModuleList()
        for i in range(5):
            self.conv_blocks.append(nn.Conv2d(filters[i], cat_channels, 3, padding=1))

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(cat_channels * 5),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Conv2d(cat_channels * 5, up_channels, 3, padding=1)
        self.final = nn.Conv2d(up_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv0(x)        # 1/1
        x1 = self.conv1(self.pool0(x0))  # 1/2
        x2 = self.conv2(self.pool1(x1))  # 1/4
        x3 = self.conv3(self.pool2(x2))  # 1/8
        x4 = self.conv4(self.pool3(x3))  # 1/16

        h, w = x0.size(2), x0.size(3)

        x0d = self.conv_blocks[0](x0)
        x1d = F.interpolate(self.conv_blocks[1](x1), size=(h, w), mode='bilinear', align_corners=True)
        x2d = F.interpolate(self.conv_blocks[2](x2), size=(h, w), mode='bilinear', align_corners=True)
        x3d = F.interpolate(self.conv_blocks[3](x3), size=(h, w), mode='bilinear', align_corners=True)
        x4d = F.interpolate(self.conv_blocks[4](x4), size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([x0d, x1d, x2d, x3d, x4d], dim=1)
        x_cat = self.bn_relu(x_cat)
        out = self.fusion_conv(x_cat)
        out = self.final(out)
        return out
