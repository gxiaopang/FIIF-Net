import torch
import torch.nn as nn
import torch.nn.functional as F
from .alignment_net import AlignmentNet
from .focus_net import FocusNet
from .fusion_net import FusionNet
from .zssr import apply_zssr_batch


class FIIFNet(nn.Module):
    """Focus Information Interaction Fusion Network (FIIF-Net).

    End-to-end framework for misaligned multi-focus image fusion.
    Three sub-networks:
    1. Alignment-Net: Optical flow-based alignment to correct jitter-induced displacements
    2. Focus-Net: Single image focus estimation for soft-cue focus prediction
    3. Fusion-Net: Dual-path interactive fusion with STA module

    Pipeline:
    1. Alignment-Net aligns all focal stack images to the first frame (reference)
    2. Focus-Net predicts focus maps from each aligned image
    3. ZSSR performs super-resolution on aligned images and focus maps
    4. Fusion-Net produces the final all-in-focus image via dual-path interactive weighting
    """
    def __init__(self, num_frames=6, in_channels=3, use_zssr=True,
                 align_hidden=64, focus_out=1, fusion_base=64,
                 zssr_iters=500, zssr_scale=2.0):
        super(FIIFNet, self).__init__()
        self.num_frames = num_frames
        self.use_zssr = use_zssr
        self.zssr_iters = zssr_iters
        self.zssr_scale = zssr_scale

        # Sub-networks
        self.alignment_net = AlignmentNet(
            in_channels=in_channels,
            hidden_dim=align_hidden,
            num_levels=4,
            num_iters=6
        )

        self.focus_net = FocusNet(
            in_channels=in_channels,
            out_channels=focus_out
        )

        self.fusion_net = FusionNet(
            img_channels=num_frames * in_channels,
            focus_channels=num_frames * focus_out,
            out_channels=in_channels,
            base_dim=fusion_base
        )

    def forward(self, focal_stack):
        """
        Args:
            focal_stack: Input focal stack [B, N, C, H, W]
                         where N is number of frames (typically 6)

        Returns:
            fused_img: All-in-focus fused image [B, C, H, W]
            aligned_imgs: Aligned images [B, N, C, H, W]
            focus_maps: Predicted focus maps [B, N, 1, H, W]
            flows: Optical flow fields [B, N-1, 2, H, W]
        """
        B, N, C, H, W = focal_stack.shape

        # Step 1: Alignment-Net
        # Use first frame as reference, align remaining frames
        ref_img = focal_stack[:, 0]  # [B, C, H, W]
        src_imgs = focal_stack[:, 1:]  # [B, N-1, C, H, W]

        aligned_src, flows = self.alignment_net(ref_img, src_imgs)

        # Concatenate reference with aligned sources
        aligned_imgs = torch.cat([ref_img.unsqueeze(1), aligned_src], dim=1)  # [B, N, C, H, W]

        # Step 2: Focus-Net - predict focus map for each aligned image
        focus_maps_list = []
        for i in range(N):
            focus_map, _, _, _ = self.focus_net(aligned_imgs[:, i])  # [B, 1, H, W]
            focus_maps_list.append(focus_map)
        focus_maps = torch.cat(focus_maps_list, dim=1)  # [B, N, H, W]

        # Reshape for Fusion-Net input
        img_stack = aligned_imgs.view(B, N * C, H, W)  # [B, N*C, H, W]

        # Step 3: ZSSR super-resolution (if enabled)
        if self.use_zssr and not self.training:
            # ZSSR is computationally expensive, typically applied during inference
            sr_imgs, sr_focus = apply_zssr_batch(
                img_stack, focus_maps,
                scale_factor=self.zssr_scale,
                num_iters=self.zssr_iters,
                device=img_stack.device
            )
        else:
            sr_imgs = None
            sr_focus = None

        # Step 4: Fusion-Net
        fused_img = self.fusion_net(img_stack, focus_maps, sr_imgs, sr_focus)

        return fused_img, aligned_imgs, focus_maps, flows

    def forward_fusion_only(self, img_stack, focus_maps, sr_imgs=None, sr_focus=None):
        """Forward pass for Fusion-Net only (used in joint training stage).

        Args:
            img_stack: [B, N*C, H, W] aligned focal stack
            focus_maps: [B, N, H, W] predicted focus maps
            sr_imgs: Optional SR images
            sr_focus: Optional SR focus maps
        """
        return self.fusion_net(img_stack, focus_maps, sr_imgs, sr_focus)

    def forward_focus_only(self, img):
        """Forward pass for Focus-Net only (used in pre-training stage)."""
        return self.focus_net(img)

    def forward_alignment_only(self, ref_img, src_imgs):
        """Forward pass for Alignment-Net only (used in pre-training stage)."""
        return self.alignment_net(ref_img, src_imgs)
