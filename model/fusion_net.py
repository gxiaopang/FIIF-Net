import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import ConvBlock, WFD, BasicBlock, UpSample, ConvRelu
from .sta import STAModule


class FusionEncoder(nn.Module):
    """Encoder for Fusion-Net.

    Two backbone blocks + one WFD unit, with LReLU and BN.
    Processes both aligned images and focus maps with identical structure.
    """
    def __init__(self, in_channels, base_dim=64):
        super(FusionEncoder, self).__init__()
        # Layer 1: ConvBlock + 2 BasicBlocks (no downsampling)
        self.layer1 = nn.Sequential(
            ConvRelu(in_channels, base_dim),
            BasicBlock(base_dim),
            BasicBlock(base_dim),
        )
        # Layer 2: BasicBlock + WFD + BasicBlock
        self.layer2 = nn.Sequential(
            BasicBlock(base_dim),
            WFD(base_dim, base_dim * 2),
            BasicBlock(base_dim * 2),
        )
        # Layer 3: BasicBlock + WFD + BasicBlock
        self.layer3 = nn.Sequential(
            BasicBlock(base_dim * 2),
            WFD(base_dim * 2, base_dim * 4),
            BasicBlock(base_dim * 4),
        )

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


class DualPathInteractive(nn.Module):
    """Dual-Path Interactive Weighting Strategy.

    Path 1: F_Ali * F_AttFoc  (focus maps as dynamic filters to highlight sharp content)
    Path 2: F_AttAli modulated by F_Foc  (image context refines focus region selection)
    """
    def __init__(self, channels):
        super(DualPathInteractive, self).__init__()
        # Attention for focus features (used in Path 1)
        self.focus_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Attention for image features (used in Path 2)
        self.img_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, f_ali, f_foc):
        """
        Args:
            f_ali: Aligned image features [B, C, H, W]
            f_foc: Focus map features [B, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        # Path 1: F_Ali * F_AttFoc
        f_att_foc = self.focus_attention(f_foc)
        path1 = f_ali * f_att_foc

        # Path 2: F_AttAli modulated by F_Foc
        f_att_ali = self.img_attention(f_ali)
        path2 = f_att_ali * f_foc

        # Fuse both paths
        out = self.fusion_conv(torch.cat([path1, path2], dim=1))
        return out


class ConvFFN(nn.Module):
    """Convolutional Feed-Forward Network after STA module."""
    def __init__(self, dim, hidden_dim=None):
        super(ConvFFN, self).__init__()
        hidden_dim = hidden_dim or dim * 4
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.lrelu(x + self.ffn(x))


class FusionNet(nn.Module):
    """Dual-Path Interactive Fusion Network (Fusion-Net).

    Four inputs: aligned focal stack images, predicted focus maps,
    and their super-resolved versions.

    Architecture:
    - Identical encoders for image and focus map
    - STA module for long-range dependency
    - Dual-path interactive weighting
    - Conv FFN
    - Multi-scale feature fusion with SR branch
    - Decoder with channel compression and element-wise addition
    - Final 3x3 conv without activation
    """
    def __init__(self, img_channels=18, focus_channels=6, out_channels=3,
                 base_dim=64, num_heads=8, grid_size=(2, 2)):
        super(FusionNet, self).__init__()

        # Encoders for original resolution
        self.img_encoder = FusionEncoder(img_channels, base_dim)
        self.focus_encoder = FusionEncoder(focus_channels, base_dim)

        # Encoders for super-resolution branch
        self.img_encoder_sr = FusionEncoder(img_channels, base_dim)
        self.focus_encoder_sr = FusionEncoder(focus_channels, base_dim)

        # STA modules for each scale
        self.sta_img_l3 = STAModule(base_dim * 4, num_heads, grid_size)
        self.sta_foc_l3 = STAModule(base_dim * 4, num_heads, grid_size)
        self.sta_img_sr_l3 = STAModule(base_dim * 4, num_heads, grid_size)
        self.sta_foc_sr_l3 = STAModule(base_dim * 4, num_heads, grid_size)

        # Dual-path interactive weighting at each scale
        self.dpi_l3 = DualPathInteractive(base_dim * 4)
        self.dpi_l2 = DualPathInteractive(base_dim * 2)
        self.dpi_l1 = DualPathInteractive(base_dim)

        # SR branch dual-path
        self.dpi_sr_l3 = DualPathInteractive(base_dim * 4)

        # Conv FFN
        self.ffn_l3 = ConvFFN(base_dim * 4)
        self.ffn_sr_l3 = ConvFFN(base_dim * 4)

        # Upsampling block for 3rd-layer encoder features (original branch)
        self.up_block_enc3 = UpSample(base_dim * 4, base_dim * 2)
        # Downsampling block for 1st-layer encoder features (SR branch)
        self.down_block_sr1 = WFD(base_dim, base_dim * 2)

        # Decoder
        self.decoder_l2 = nn.Sequential(
            BasicBlock(base_dim * 2),
            UpSample(base_dim * 2, base_dim),
            BasicBlock(base_dim),
        )
        self.decoder_l1 = nn.Sequential(
            BasicBlock(base_dim),
            BasicBlock(base_dim),
        )

        # Channel compression for final fusion
        self.compress = nn.Conv2d(base_dim * 2, base_dim, kernel_size=1, bias=False)

        # Final output: 3x3 conv without activation
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, stride=1,
                                    padding=1, padding_mode='reflect')

    def forward(self, img_stack, focus_maps, sr_imgs=None, sr_focus=None):
        """
        Args:
            img_stack: Aligned focal stack images [B, 18, H, W]
            focus_maps: Predicted focus maps [B, 6, H, W]
            sr_imgs: Super-resolved focal stack images [B, 18, H, W] (optional)
            sr_focus: Super-resolved focus maps [B, 6, H, W] (optional)
        Returns:
            Fused image [B, 3, H, W]
        """
        # Encode original resolution
        f1_img, f2_img, f3_img = self.img_encoder(img_stack)
        f1_foc, f2_foc, f3_foc = self.focus_encoder(focus_maps)

        # STA on deepest features
        f3_img_att = self.sta_img_l3(f3_img)
        f3_foc_att = self.sta_foc_l3(f3_foc)

        # Dual-path interactive at scale 3
        f3_fused = self.dpi_l3(f3_img_att, f3_foc_att)

        # Conv FFN
        f3_fused = self.ffn_l3(f3_fused)

        # SR branch processing
        if sr_imgs is not None and sr_focus is not None:
            f1_img_sr, f2_img_sr, f3_img_sr = self.img_encoder_sr(sr_imgs)
            f1_foc_sr, f2_foc_sr, f3_foc_sr = self.focus_encoder_sr(sr_focus)

            # STA on SR features
            f3_img_sr_att = self.sta_img_sr_l3(f3_img_sr)
            f3_foc_sr_att = self.sta_foc_sr_l3(f3_foc_sr)

            # Dual-path interactive on SR at scale 3
            f3_sr_fused = self.dpi_sr_l3(f3_img_sr_att, f3_foc_sr_att)
            f3_sr_fused = self.ffn_sr_l3(f3_sr_fused)

            # Multi-scale fusion: upsample original 3rd-layer, downsample SR 1st-layer
            f3_up = self.up_block_enc3(f3_fused)
            f1_sr_down = self.down_block_sr1(f1_img_sr)

            # Combine
            f2_combined = f3_up + f2_img + f2_foc + f1_sr_down
        else:
            # Without SR branch
            f3_up = self.up_block_enc3(f3_fused)
            f2_combined = f3_up + f2_img + f2_foc

        # Dual-path interactive at scale 2
        f2_fused = self.dpi_l2(f2_img, f2_foc)
        f2_combined = f2_combined + f2_fused

        # Decoder layer 2
        d2 = self.decoder_l2(f2_combined)

        # Dual-path interactive at scale 1
        f1_fused = self.dpi_l1(f1_img, f1_foc)

        # Combine with decoder output
        f1_combined = d2 + f1_fused

        # Decoder layer 1
        d1 = self.decoder_l1(f1_combined)

        # Channel compression + element-wise addition
        out = self.compress(torch.cat([d1, f1_fused], dim=1))

        # Final 3x3 conv without activation
        out = self.final_conv(out)

        return out
