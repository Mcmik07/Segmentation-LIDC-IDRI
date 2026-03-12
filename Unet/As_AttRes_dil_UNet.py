import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Bloc résiduel dilaté
#   - conv 3x3 (d=1) -> conv 3x3 dilatée (d>1) + skip
# ======================================================
class ResDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.drop  = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.proj  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) \
                     if in_channels != out_channels else nn.Identity()
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))

        out = out + identity
        return self.act_out(out)


# =================================================================
# DoubleConv MODIFIÉ :
#  conv 3x3 "normale"  ->  ResDilatedBlock (2e conv dilatée + skip)
# =================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=2, dropout_p=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.act1  = nn.ReLU(inplace=True)

        self.res_dilated = ResDilatedBlock(mid_channels, out_channels,
                                           dilation=dilation, dropout_p=dropout_p)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.res_dilated(x)
        return x


# =========================
# Spatial Attention (CBAM-like)
# =========================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv(x_att)
        att_map = self.sigmoid(x_att)
        return x * att_map


# =========================
# Attention Gate (Oktay et al.)
# =========================
class AttentionGate(nn.Module):
    def __init__(self, skip_c, gate_c, inter_c):
        super().__init__()
        self.theta_x = nn.Conv2d(skip_c, inter_c, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv2d(gate_c, inter_c, kernel_size=1, bias=False)
        self.psi     = nn.Conv2d(inter_c, 1, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm2d(inter_c)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _pad_or_crop(tensor, ref):
        diffY = tensor.size(2) - ref.size(2)
        diffX = tensor.size(3) - ref.size(3)
        if diffY == 0 and diffX == 0:
            return tensor
        if diffY >= 0 and diffX >= 0:
            return tensor[:, :, diffY//2:tensor.size(2)-((diffY+1)//2),
                                diffX//2:tensor.size(3)-((diffX+1)//2)]
        padY1 = (-diffY)//2 if diffY < 0 else 0
        padY2 = (-diffY) - padY1 if diffY < 0 else 0
        padX1 = (-diffX)//2 if diffX < 0 else 0
        padX2 = (-diffX) - padX1 if diffX < 0 else 0
        return F.pad(tensor, (padX1, padX2, padY1, padY2))

    def forward(self, x_skip, g):
        g = self._pad_or_crop(g, x_skip)
        theta_x = self.theta_x(x_skip)
        phi_g   = self.phi_g(g)
        f = self.relu(self.bn(theta_x + phi_g))
        psi = self.sigmoid(self.psi(f))
        return x_skip * psi


# =========================
# ASPP (DeepLabV3-style)
# =========================
class ASPP(nn.Module):
    """
    Branches: 1x1, 3x3@r for r in rates, (optional) image pooling
    Concat -> 1x1 projection
    """
    def __init__(self, in_channels, out_channels=None, rates=(1,2,3), use_gap=True, dropout_p=0.0):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_c = max(32, in_channels // 4)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_c),
            nn.ReLU(inplace=True)
        )

        self.branchs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, inter_c, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(inter_c),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])

        self.use_gap = use_gap
        if use_gap:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, inter_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_c),
                nn.ReLU(inplace=True)
            )

        total_c = inter_c * (1 + len(self.branchs) + (1 if use_gap else 0))
        self.project = nn.Sequential(
            nn.Conv2d(total_c, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.branch1(x)]
        feats += [b(x) for b in self.branchs]
        if self.use_gap:
            gp = self.image_pool(x)
            gp = F.interpolate(gp, size=(h, w), mode='bilinear', align_corners=False)
            feats.append(gp)
        y = torch.cat(feats, dim=1)
        return self.project(y)


# =========================================
# Down: MaxPool2d -> DoubleConv (modifié)
# =========================================
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# ======================================================
# Up: Upsample/Deconv -> AttentionGate(skip,g) -> concat -> DoubleConv
# ======================================================
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.0):
        """
        in_channels : nb de canaux APRÈS concat (skip + up), passé à DoubleConv
        out_channels: nb de canaux de sortie du bloc Up
        """
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_p=dropout)
            up_c = in_channels // 2
            skip_c = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout)
            up_c = in_channels // 2
            skip_c = in_channels // 2

        inter_c = max(1, skip_c // 2)
        self.ag = AttentionGate(skip_c=skip_c, gate_c=up_c, inter_c=inter_c)

    @staticmethod
    def _pad_for_concat(x_small, x_ref):
        diffY = x_ref.size(2) - x_small.size(2)
        diffX = x_ref.size(3) - x_small.size(3)
        if diffY == 0 and diffX == 0:
            return x_small
        return F.pad(
            x_small,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

    def forward(self, x_dec, x_enc):
        x_up = self.up(x_dec)
        x_up = self._pad_for_concat(x_up, x_enc)
        x_enc_att = self.ag(x_enc, x_up)   # skip filtré
        x = torch.cat([x_enc_att, x_up], dim=1)
        return self.conv(x)


# =========================
# Out conv 1x1
# =========================
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# =========================
#     AttRes_dil_UNet
# =========================
class As_AttRes_dil_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1
        bottleneck_c = 1024 // factor
        self.down4 = Down(512, bottleneck_c)  # bottleneck in

        # Bottleneck: ASPP -> Spatial Attention
        self.aspp = ASPP(in_channels=bottleneck_c,
                         out_channels=bottleneck_c,
                         rates=(1, 2, 3),   # à tester: (1,3,5)
                         use_gap=True,
                         dropout_p=0.0)
        self.spatial_att = SpatialAttention(kernel_size=7)

        # Decoder (AG intégrés dans Up)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128  // factor, bilinear)
        self.up4 = Up(128,  64, bilinear)

        # Sortie
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck
        x5 = self.aspp(x5)          # contexte multi-échelle
        x5 = self.spatial_att(x5)   # focus spatial

        # Décodeur
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        logits = self.outc(x)
        return logits



model = As_AttRes_dil_UNet(n_channels=1, n_classes=1)

# Total paramètres
total_params = sum(p.numel() for p in model.parameters())

# Paramètres entraînables seulement
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total params:", total_params)
print("Trainable params:", trainable_params)