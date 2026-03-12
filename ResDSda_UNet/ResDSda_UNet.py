import torch
import torch.nn as nn
import torch.nn.functional as F


# Double convolutional block (conv => [BN] => ReLU) * 2
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# Residual Block for Deep Residual DO-Conv Layer
class ResDS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)) + x)  # Residual connection


# Channel Attention (CCA)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


# Spatial Attention (CSA)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, 1, keepdim=True)
        max_pool, _ = torch.max(x, 1, keepdim=True)
        x_out = torch.cat([avg_pool, max_pool], dim=1)
        x_out = self.conv(x_out)
        return x * self.sigmoid(x_out)


# Dense Atrous Spatial Pyramid Pooling (DASPP)
class DASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=4, padding=4)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.conv5(x)
        return out1 + out2 + out3 + out4 + out5


# Downscaling block: maxpool followed by a double convolution
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Upscaling block: bilinear upsampling followed by a double convolution
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.3):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # Attention layers
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        # Apply attention mechanisms
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return self.conv(x)


# Output convolutional block
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# U-Net model with the fusion of the convolutional blocks
class ResDSda_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResDSda_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial double convolution
        self.inc = DoubleConv(n_channels, 32)
        # Downscaling blocks
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # Adding DASPP
        self.daspp = DASPP(512 // factor, 512 // factor)

        # Upscaling blocks
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)

        # Final output convolution
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply DASPP after downscaling
        x5 = self.daspp(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        logits = self.outc(x)
        return logits
