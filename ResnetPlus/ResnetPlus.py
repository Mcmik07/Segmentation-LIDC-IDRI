import torch
import torch.nn as nn
import torch.nn.functional as F


# Bloc résiduel
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# Bloc de l'encodeur de ResUNet++
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.residual_block(x)
        pooled = self.maxpool(out)
        return out, pooled


# Bloc du décodeur de ResUNet++
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.residual_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        upsampled = self.upconv(x)
        concat = torch.cat([upsampled, skip_connection], dim=1)
        return self.residual_block(concat)


# Modèle ResUNet++
class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNetPlusPlus, self).__init__()

        # Encodeur
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Goulot (Bottleneck)
        self.bottleneck = ResidualBlock(512, 1024)

        # Décodeur
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)

        # Sortie
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encodeur
        enc1, pool1 = self.encoder1(x)
        enc2, pool2 = self.encoder2(pool1)
        enc3, pool3 = self.encoder3(pool2)
        enc4, pool4 = self.encoder4(pool3)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Décodeur
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Sortie
        out = self.final_conv(dec1)
        return out
