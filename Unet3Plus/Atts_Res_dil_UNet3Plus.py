import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- ResDilated ---------------------
class ResDilatedBlock(nn.Module):
    """conv 3x3 (d=1) -> conv 3x3 dilated (d>1) + skip"""
    def __init__(self, in_channels, out_channels, dilation=2, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.drop  = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self.proj  = nn.Conv2d(in_channels, out_channels, 1, bias=False) \
                     if in_channels != out_channels else nn.Identity()
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        out = out + identity
        return self.act_out(out)

# ------------- DoubleConv version ResDil --------------
class VGGBlock(nn.Module):
    """conv 3x3 -> BN -> ReLU -> ResDilated (encoder version)"""
    def __init__(self, in_channels, middle_channels, out_channels, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(middle_channels)
        self.act1  = nn.ReLU(inplace=True)
        # on passe middle -> out via ResDilated
        self.resd  = ResDilatedBlock(middle_channels, out_channels, dilation=dilation)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.resd(x)
        return x

# -------- Decoder block: proj -> ResDilated ----------
class DecProjResDil(nn.Module):
    """Projeter chaque échelle vers cat_channels puis ResDilated (décodeur)"""
    def __init__(self, in_channels, cat_channels, dilation=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, cat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cat_channels),
            nn.ReLU(inplace=True),
        )
        self.resd = ResDilatedBlock(cat_channels, cat_channels, dilation=dilation)

    def forward(self, x):
        x = self.proj(x)
        x = self.resd(x)
        return x

# ------------------ Spatial Attention ----------------
class SpatialAttention(nn.Module):
    """CBAM spatial: AvgPool_c + MaxPool_c -> conv kxk -> sigmoid"""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

# ---------------------- UNet3+ -----------------------
class Atts_Res_dil_UNet3Plus(nn.Module):
    def __init__(self, num_classes=1, input_channels=1, feature_scale=1, dilation=2):
        super().__init__()
        filters = [32, 64, 126, 256, 512]               # (reprend ton choix)
        filters = [int(x / feature_scale) for x in filters]

        # Encoder (Conv3x3 -> ResDilated)
        self.conv0 = VGGBlock(input_channels, filters[0], filters[0], dilation=dilation)
        self.pool0 = nn.MaxPool2d(2)
        self.conv1 = VGGBlock(filters[0], filters[1], filters[1], dilation=dilation)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = VGGBlock(filters[1], filters[2], filters[2], dilation=dilation)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = VGGBlock(filters[2], filters[3], filters[3], dilation=dilation)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = VGGBlock(filters[3], filters[4], filters[4], dilation=dilation)

        # Spatial attention AU BOTTLENECK (sur x4)
        self.spatial_att = SpatialAttention(kernel_size=7)

        # Decoder (chaque échelle: proj -> ResDilated -> upsample)
        cat_channels = filters[0]
        up_channels  = filters[0]

        self.dec_blocks = nn.ModuleList([
            DecProjResDil(filters[i], cat_channels, dilation=dilation) for i in range(5)
        ])

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(cat_channels * 5),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Conv2d(cat_channels * 5, up_channels, 3, padding=1)
        self.final = nn.Conv2d(up_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # ---------- Encoder ----------
        x0 = self.conv0(x)                 # 1/1
        x1 = self.conv1(self.pool0(x0))    # 1/2
        x2 = self.conv2(self.pool1(x1))    # 1/4
        x3 = self.conv3(self.pool2(x2))    # 1/8
        x4 = self.conv4(self.pool3(x3))    # 1/16

        # Spatial attention au bottleneck
        x4 = self.spatial_att(x4)

        # ---------- Decoder (UNet3+ style concat multi-échelle) ----------
        h, w = x0.size(2), x0.size(3)

        x0d = self.dec_blocks[0](x0)
        x1d = F.interpolate(self.dec_blocks[1](x1), size=(h, w), mode='bilinear', align_corners=True)
        x2d = F.interpolate(self.dec_blocks[2](x2), size=(h, w), mode='bilinear', align_corners=True)
        x3d = F.interpolate(self.dec_blocks[3](x3), size=(h, w), mode='bilinear', align_corners=True)
        x4d = F.interpolate(self.dec_blocks[4](x4), size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([x0d, x1d, x2d, x3d, x4d], dim=1)
        x_cat = self.bn_relu(x_cat)
        out = self.fusion_conv(x_cat)
        out = self.final(out)
        return out
