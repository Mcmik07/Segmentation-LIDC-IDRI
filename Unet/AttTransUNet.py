import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Building Blocks (shared)
# -----------------------------
class ConvBlock(nn.Module):
    """Two 3x3 convs + BN + ReLU ("same" padding)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upsample by 2 then 3x3 conv + BN + ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    """Attention gate for skip connections (additive attention)."""
    def __init__(self, F_g, F_l, n_coefficients):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class TransformerBlock(nn.Module):
    """A simple ViT-style encoder block applied at the bottleneck feature map.
    Expects input reshaped to (L, B, C) where L = H*W, C = channels.
    """
    def __init__(self, dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim),
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)

    def forward(self, x):  # x: (L, B, C)
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x


# -----------------------------
# Fused Architecture
# -----------------------------
class AttnTransUNet(nn.Module):
    """
    Fusion of Attention U-Net (attention-gated skips + hierarchical decoder)
    and TransUNet-style Transformer at the bottleneck.

    - Encoder: 5 stages with ConvBlock and MaxPool.
    - Bottleneck: TransformerBlock operates over flattened spatial tokens.
    - Decoder: Classic Attention U-Net up-path with attention-gated skip connections.
    - Head: 1x1 conv to num_classes. Optionally an extra multi-scale fusion head
            (disabled by default; can be enabled with use_multiscale_fusion=True).

    Args:
        input_channels (int): number of input channels (e.g., 1 for CT slices).
        num_classes (int): output channels for segmentation (1 for binary / BCEWithLogits).
        base_filters (int): base number of filters at stage 1 (default 32).
        num_heads (int): transformer heads at bottleneck.
        ff_dim (int): transformer FFN inner dim.
        use_multiscale_fusion (bool): if True, adds a lightweight UNet3++-like
            multi-scale fusion head that concatenates projected encoder features
            and the transformed bottleneck, then fuses to produce the final logits.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 32,
        num_heads: int = 8,
        ff_dim: int = 512,
        use_multiscale_fusion: bool = False,
    ):
        super().__init__()

        f1 = base_filters
        f2 = f1 * 2
        f3 = f2 * 2
        f4 = f3 * 2
        f5 = f4 * 2
        filters = [f1, f2, f3, f4, f5]

        # Encoder
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(input_channels, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.enc3 = ConvBlock(f2, f3)
        self.enc4 = ConvBlock(f3, f4)
        self.enc5 = ConvBlock(f4, f5)

        # Bottleneck Transformer (tokenize HxW locations; channel = f5)
        self.transformer = TransformerBlock(dim=f5, num_heads=num_heads, ff_dim=ff_dim)

        # Decoder (Attention U-Net style)
        self.up5 = UpConv(f5, f4)
        self.att5 = AttentionBlock(F_g=f4, F_l=f4, n_coefficients=f4 // 2)
        self.dec5 = ConvBlock(f4 + f4, f4)

        self.up4 = UpConv(f4, f3)
        self.att4 = AttentionBlock(F_g=f3, F_l=f3, n_coefficients=f3 // 2)
        self.dec4 = ConvBlock(f3 + f3, f3)

        self.up3 = UpConv(f3, f2)
        self.att3 = AttentionBlock(F_g=f2, F_l=f2, n_coefficients=f2 // 2)
        self.dec3 = ConvBlock(f2 + f2, f2)

        self.up2 = UpConv(f2, f1)
        self.att2 = AttentionBlock(F_g=f1, F_l=f1, n_coefficients=f1 // 2)
        self.dec2 = ConvBlock(f1 + f1, f1)

        # Heads
        self.out_head = nn.Conv2d(f1, num_classes, kernel_size=1, stride=1, padding=0)

        # Optional multi-scale fusion (UNet3++-like) using encoder maps + transformed bottleneck
        self.use_multiscale_fusion = use_multiscale_fusion
        if use_multiscale_fusion:
            proj_c = f1  # project each scale to base width
            self.proj_layers = nn.ModuleList([
                nn.Conv2d(f1, proj_c, 3, padding=1),  # e1
                nn.Conv2d(f2, proj_c, 3, padding=1),  # e2
                nn.Conv2d(f3, proj_c, 3, padding=1),  # e3
                nn.Conv2d(f4, proj_c, 3, padding=1),  # e4
                nn.Conv2d(f5, proj_c, 3, padding=1),  # transformed e5
            ])
            self.bn_relu = nn.Sequential(
                nn.BatchNorm2d(proj_c * 5),
                nn.ReLU(inplace=True),
            )
            self.fusion_conv = nn.Conv2d(proj_c * 5, f1, 3, padding=1)
            self.out_head_fusion = nn.Conv2d(f1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)            # 1/1
        e2 = self.enc2(self.maxpool(e1))  # 1/2
        e3 = self.enc3(self.maxpool(e2))  # 1/4
        e4 = self.enc4(self.maxpool(e3))  # 1/8
        e5 = self.enc5(self.maxpool(e4))  # 1/16

        # Bottleneck Transformer (flatten spatial, apply MHA, reshape back)
        B, C, H, W = e5.shape
        tokens = e5.flatten(2).permute(2, 0, 1)  # (L, B, C) where L=H*W
        tokens = self.transformer(tokens)
        e5_t = tokens.permute(1, 2, 0).reshape(B, C, H, W)

        # Decoder with attention-gated skips
        d5 = self.up5(e5_t)
        s4 = self.att5(gate=d5, skip_connection=e4)
        d5 = torch.cat([s4, d5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        s3 = self.att4(gate=d4, skip_connection=e3)
        d4 = torch.cat([s3, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        s2 = self.att3(gate=d3, skip_connection=e2)
        d3 = torch.cat([s2, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        s1 = self.att2(gate=d2, skip_connection=e1)
        d2 = torch.cat([s1, d2], dim=1)
        d2 = self.dec2(d2)  # highest-res decoder feature

        out_main = self.out_head(d2)

        if not self.use_multiscale_fusion:
            return out_main

        # Optional multi-scale fusion head (UNet3++-like with encoder scales + transformed bottleneck)
        h, w = e1.size(2), e1.size(3)
        p_e1 = self.proj_layers[0](e1)
        p_e2 = F.interpolate(self.proj_layers[1](e2), size=(h, w), mode='bilinear', align_corners=True)
        p_e3 = F.interpolate(self.proj_layers[2](e3), size=(h, w), mode='bilinear', align_corners=True)
        p_e4 = F.interpolate(self.proj_layers[3](e4), size=(h, w), mode='bilinear', align_corners=True)
        p_e5 = F.interpolate(self.proj_layers[4](e5_t), size=(h, w), mode='bilinear', align_corners=True)

        x_cat = torch.cat([p_e1, p_e2, p_e3, p_e4, p_e5], dim=1)
        x_cat = self.bn_relu(x_cat)
        fused = self.fusion_conv(x_cat)
        out_fusion = self.out_head_fusion(fused)

        # Return both heads to allow flexible losses (deep supervision / auxiliary loss)
        return out_main, out_fusion


if __name__ == "__main__":
    # quick smoke test
    model = AttnTransUNet(input_channels=1, num_classes=1, base_filters=32, num_heads=8, ff_dim=512, use_multiscale_fusion=False)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    if isinstance(y, tuple):
        out_main, out_fusion = y
        print("out_main:", out_main.shape, "out_fusion:", out_fusion.shape)
    else:
        print("out:", y.shape)
