import torch
import torch.nn as nn
import torch.nn.functional as F


# Transformer Block : Un bloc Transformer pour capturer les relations globales
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


# Double convolutional block (conv => [BN] => ReLU) * 2
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
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


# Downscaling block: maxpool followed by a double convolution
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Bloc Attention
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)  # Traitement du signal de gating
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)  # Traitement de la skip connection
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)  # Sortie de l'attention (1 canal)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: skip connection, g: gating signal (from decoder)
        x1 = self.W_x(x)
        g1 = self.W_g(g)
        psi = self.sigmoid(self.psi(F.relu(x1 + g1)))  # Carte d'attention
        return x * psi  # Appliquer la carte d'attention à la skip connection


# Bloc d'Upscaling avec Attention
class UpWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, gating_channels, inter_channels, bilinear=True):
        super(UpWithAttention, self).__init__()

        # Upsampling layer (bilinéaire ou transposé)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Attention Gate
        self.attention_gate = AttentionGate(in_channels // 2, gating_channels, inter_channels)

        # Convolution après concaténation
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x1: feature map du décodeur, x2: skip connection de l'encodeur
        x1 = self.up(x1)  # Upsample du signal de décodeur

        # Appliquer l'Attention Gate à la skip connection
        x2 = self.attention_gate(x2,
                                 x1)  # Appliquer l'attention sur la skip connection (x2) avec le signal de gating (x1)

        # Concatenation après attention
        x = torch.cat([x1, x2], dim=1)  # Concaténer sur la dimension des canaux
        return self.double_conv(x)  # Appliquer la convolution après concaténation


# Output convolutional block
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# U-Net avec Attention Gates et Transformer intégré
class UNetTransWithAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, num_heads=8, ff_dim=512):
        super(UNetTransWithAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial double convolution
        self.inc = DoubleConv(n_channels, 32)

        # Blocs de Downscaling
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # Transformer Block pour capter les dépendances globales
        self.transformer = TransformerBlock(dim=512 // factor, num_heads=num_heads, ff_dim=ff_dim)

        # Blocs d'Upscaling avec attention
        self.up1 = UpWithAttention(512, 256 // factor, 256, 64, bilinear)
        self.up2 = UpWithAttention(256, 128 // factor, 128, 32, bilinear)
        self.up3 = UpWithAttention(128, 64 // factor, 64, 16, bilinear)
        self.up4 = UpWithAttention(64, 32, 32, 8, bilinear)

        # Convolution de sortie
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Appliquer le Transformer sur la dernière couche de l'encodeur
        transformer_input = x5.flatten(2).permute(2, 0, 1)  # Reshaper pour le Transformer
        x5 = self.transformer(transformer_input).permute(1, 2, 0).reshape(-1, 512 // 2, x5.size(2), x5.size(3))

        # Décodeur avec attention appliquée sur les skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Sortie finale
        logits = self.outc(x)
        return logits
