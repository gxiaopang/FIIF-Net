import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ConvBlock, WFD, UpSample


class FocusNet(nn.Module):
    """Single Image Focus Estimation Network (Focus-Net).

    Encoder-decoder architecture:
    - Encoder: 4 ConvBlocks + 3 WFD downsampling units
    - MHSA at deepest layer
    - Decoder with skip connections
    - Auxiliary focus map predictions at 1/4 and 1/2 scale
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(FocusNet, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(in_channels, 32)     # 600 -> 600
        self.down1 = WFD(32, 32)                     # 600 -> 300
        self.conv2 = ConvBlock(32, 64)                # 300 -> 300
        self.down2 = WFD(64, 64)                      # 300 -> 150
        self.conv3 = ConvBlock(64, 128)               # 150 -> 150
        self.down3 = WFD(128, 128)                    # 150 -> 75
        self.conv4 = ConvBlock(128, 256)              # 75 -> 75

        # Multi-Head Self-Attention at deepest layer
        self.mhsa = nn.MultiheadAttention(256, 8, batch_first=False)

        # Decoder
        self.up1 = UpSample(256, 128)                 # 75 -> 150
        self.conv5 = ConvBlock(256, 128)

        self.up2 = UpSample(128, 64)                  # 150 -> 300
        self.conv6 = ConvBlock(128, 64)

        self.up3 = UpSample(64, 32)                   # 300 -> 600
        self.conv7 = ConvBlock(64, 32)

        # Auxiliary focus map branch at 1/4 scale (150x150)
        self.aux1_branch = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Auxiliary focus map branch at 1/2 scale (300x300)
        self.aux2_branch = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Auxiliary focus map branch at full scale (600x600)
        self.aux3_branch = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # Final output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)          # [B, 32, 600, 600]
        d1 = self.down1(e1)         # [B, 32, 300, 300]
        e2 = self.conv2(d1)         # [B, 64, 300, 300]
        d2 = self.down2(e2)         # [B, 64, 150, 150]
        e3 = self.conv3(d2)         # [B, 128, 150, 150]
        d3 = self.down3(e3)         # [B, 128, 75, 75]
        e4 = self.conv4(d3)         # [B, 256, 75, 75]

        # MHSA
        B, C, H, W = e4.shape
        e4_flat = e4.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]
        e4_att, _ = self.mhsa(e4_flat, e4_flat, e4_flat)
        e4_att = e4_att.permute(1, 2, 0).view(B, C, H, W)

        # Decoder with skip connections
        u1 = self.up1(e4_att)                              # [B, 128, 150, 150]
        cat1 = torch.cat([u1, e3], dim=1)                  # [B, 256, 150, 150]
        dec1 = self.conv5(cat1)                            # [B, 128, 150, 150]

        # Auxiliary prediction at 1/4 scale
        aux1 = self.aux1_branch(dec1)                      # [B, 1, 150, 150]

        u2 = self.up2(dec1)                                # [B, 64, 300, 300]
        cat2 = torch.cat([u2, e2], dim=1)                  # [B, 128, 300, 300]
        dec2 = self.conv6(cat2)                            # [B, 64, 300, 300]

        # Auxiliary prediction at 1/2 scale
        aux2 = self.aux2_branch(dec2)                      # [B, 1, 300, 300]

        u3 = self.up3(dec2)                                # [B, 32, 600, 600]
        cat3 = torch.cat([u3, e1], dim=1)                  # [B, 64, 600, 600]
        dec3 = self.conv7(cat3)                            # [B, 32, 600, 600]

        # Auxiliary prediction at full scale
        aux3 = self.aux3_branch(dec3)                      # [B, 1, 600, 600]

        # Final output
        out = self.sigmoid(self.final_conv(dec3))           # [B, 1, 600, 600]

        return out, aux1, aux2, aux3
