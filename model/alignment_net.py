import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import WFD, ConvBlock


class RFDPBlock(nn.Module):
    """Residual Feature Downsampling Pyramid Block.

    Fuses WFD and MaxPooling via residual connections with a learnable Repair Factor.
    Equation: RF^{l+1}_{Out} = Conv(MP(RF^l_In) * RF^{l+1}_In + RF^{l+1}_In)
    """
    def __init__(self, in_channels, out_channels):
        super(RFDPBlock, self).__init__()
        self.wfd = WFD(in_channels, out_channels)
        self.repair_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                     padding=1, padding_mode='reflect', bias=False)
        self.repair_bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # WFD branch output
        wfd_out = self.wfd(x)
        # Max pooling branch
        mp_out = F.adaptive_max_pool2d(x, output_size=wfd_out.shape[2:])
        # Repair factor: Conv(MP(x) * WFD(x) + WFD(x))
        repair = self.repair_conv(mp_out * wfd_out + wfd_out)
        repair = self.lrelu(self.repair_bn(repair))
        return repair


class FeatureEncoder(nn.Module):
    """Feature extraction with RFDP pyramid, as described in the paper.

    Builds a multi-scale feature pyramid using RFDP blocks.
    """
    def __init__(self, in_channels=3, hidden_dim=64, num_levels=4):
        super(FeatureEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # First layer: ConvBlock + RFDP
        self.layers.append(nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            RFDPBlock(hidden_dim, hidden_dim)
        ))

        # Subsequent layers
        for i in range(1, num_levels):
            in_ch = hidden_dim * (2 ** (i - 1))
            out_ch = hidden_dim * (2 ** i)
            self.layers.append(nn.Sequential(
                ConvBlock(in_ch, out_ch),
                RFDPBlock(out_ch, out_ch)
            ))

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class ContextEncoder(nn.Module):
    """Context encoder that captures global context priors from the first frame."""
    def __init__(self, in_channels=3, hidden_dim=64, context_dim=128):
        super(ContextEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, hidden_dim),
            WFD(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, context_dim),
            WFD(context_dim, context_dim),
            ConvBlock(context_dim, context_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class CorrelationPyramid(nn.Module):
    """Multi-scale correlation volume pyramid.

    Computes 4D correlation volume and applies multi-scale pooling.
    """
    def __init__(self, num_levels=4):
        super(CorrelationPyramid, self).__init__()
        self.num_levels = num_levels

    def forward(self, feat1, feat2):
        """Compute correlation pyramid between two feature maps."""
        B, C, H, W = feat1.shape
        # Reshape for matrix multiplication
        feat1 = feat1.view(B, C, H * W)
        feat2 = feat2.view(B, C, H * W)
        # Compute correlation volume
        corr = torch.bmm(feat1.transpose(1, 2), feat2)  # [B, H*W, H*W]
        corr = corr.view(B, H, W, H, W)
        corr = corr / (C ** 0.5)

        # Build pyramid (pool only src dims, keep ref dims at full resolution)
        pyramid = [corr]
        src_h, src_w = H, W  # track src spatial dims
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr.view(B, H * W, src_h, src_w), kernel_size=2, stride=2)
            src_h, src_w = corr.shape[2:]
            corr = corr.view(B, H, W, src_h, src_w)
            pyramid.append(corr)

        return pyramid


class DLO(nn.Module):
    """Dynamic Lookup Operator.

    Dynamically adjusts the receptive field based on input features.
    Computes expansion coefficients (αx, αy in [2,4]) and offset factors (dx, dy in [-2,2]).
    p_new = (αx*u + f(u)*dx, αy*v + f(v)*dy)
    """
    def __init__(self, in_channels, num_levels=4, out_dim=256):
        super(DLO, self).__init__()
        self.num_levels = num_levels
        self.out_dim = out_dim
        # Project 2-channel flow to in_channels for parameter prediction
        self.flow_proj = nn.Sequential(
            nn.Conv2d(2, in_channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Per-level parameters: αx, αy, dx, dy
        self.param_nets = nn.ModuleList()
        for _ in range(num_levels):
            self.param_nets.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, 4),
            ))
        # Per-level projection to fixed output dim
        self.level_projs = nn.ModuleList()
        for _ in range(num_levels):
            self.level_projs.append(nn.Sequential(
                nn.Conv2d(1, out_dim, kernel_size=1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
            ))

    def adaptive_func(self, x):
        """f(·) constraining output to [-1, 1] for geometric symmetry."""
        return torch.tanh(x)

    def forward(self, corr_pyramid, flow):
        """Apply DLO to dynamically look up from correlation pyramid.

        Correlation volume is 5D: [B, H_ref, W_ref, H_src, W_src].
        At deeper pyramid levels, H_src < H_ref due to pooling.
        For each reference pixel, we look up in the source correlation map
        at the position indicated by the (scaled) flow.
        """
        B = flow.shape[0]
        lookup_results = []

        # Project flow to feature dim for parameter prediction
        flow_feat = self.flow_proj(flow)

        # Reference spatial dims (same across all levels)
        H_ref, W_ref = corr_pyramid[0].shape[2], corr_pyramid[0].shape[3]

        for level in range(self.num_levels):
            corr = corr_pyramid[level]
            H_src, W_src = corr.shape[-2], corr.shape[-1]

            # Get dynamic parameters from projected flow features
            params = self.param_nets[level](flow_feat)
            alpha_x = 2.0 + 2.0 * torch.sigmoid(params[:, 0])  # [2, 4]
            alpha_y = 2.0 + 2.0 * torch.sigmoid(params[:, 1])  # [2, 4]
            d_x = 2.0 * torch.tanh(params[:, 2])  # [-2, 2]
            d_y = 2.0 * torch.tanh(params[:, 3])  # [-2, 2]

            # Reshape for broadcasting with [B, H_ref, W_ref] flow
            alpha_x = alpha_x.view(B, 1, 1)
            alpha_y = alpha_y.view(B, 1, 1)
            d_x = d_x.view(B, 1, 1)
            d_y = d_y.view(B, 1, 1)

            # Scale flow to reference feature resolution
            flow_ref = F.interpolate(flow, size=(H_ref, W_ref), mode='bilinear', align_corners=False)
            flow_ref = flow_ref.permute(0, 2, 3, 1)  # [B, H_ref, W_ref, 2]

            # Compute new sampling points (in source pixel coords)
            u = flow_ref[:, :, :, 0]  # [B, H_ref, W_ref]
            v = flow_ref[:, :, :, 1]
            f_u = self.adaptive_func(u)
            f_v = self.adaptive_func(v)

            p_new_x = alpha_x * u + f_u * d_x
            p_new_y = alpha_y * v + f_v * d_y

            # Scale coordinates from ref resolution to src resolution at this level
            scale_x = W_src / W_ref
            scale_y = H_src / H_ref
            p_new_x = p_new_x * scale_x
            p_new_y = p_new_y * scale_y

            # Normalize to [-1, 1] for grid_sample (in src coordinate space)
            grid_x = 2.0 * p_new_x / max(W_src - 1, 1) - 1.0
            grid_y = 2.0 * p_new_y / max(H_src - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H_ref, W_ref, 2]

            # Reshape corr: [B, H_ref*W_ref, H_src, W_src] — each ref pixel is a "channel"
            corr_flat = corr.view(B, H_ref * W_ref, H_src, W_src)
            # grid_sample: input [B, C, H_in, W_in], grid [B, H_out, W_out, 2]
            sampled = F.grid_sample(corr_flat, grid, mode='bilinear', padding_mode='zeros',
                                    align_corners=False)
            # sampled: [B, H_ref*W_ref, H_ref, W_ref]
            # Extract per-pixel correlation: for ref pixel (i,j), take channel i*W+j at position (i,j)
            i_idx = torch.arange(H_ref, device=sampled.device).view(1, H_ref, 1)
            j_idx = torch.arange(W_ref, device=sampled.device).view(1, 1, W_ref)
            ch_idx = i_idx * W_ref + j_idx  # [1, H_ref, W_ref]
            b_idx = torch.arange(B, device=sampled.device).view(B, 1, 1)
            sampled_per_pixel = sampled[b_idx, ch_idx, i_idx, j_idx]  # [B, H_ref, W_ref]
            sampled_per_pixel = sampled_per_pixel.unsqueeze(1)  # [B, 1, H_ref, W_ref]
            # Project to fixed output dim
            projected = self.level_projs[level](sampled_per_pixel)  # [B, out_dim, H_ref, W_ref]
            lookup_results.append(projected)

        # Concatenate multi-level lookups: all at ref resolution with fixed channels
        result = torch.cat(lookup_results, dim=1)  # [B, num_levels*out_dim, H_ref, W_ref]
        return result


class UpdateBlock(nn.Module):
    """GRU-based iterative update module with DLO."""
    def __init__(self, corr_dim, flow_dim=128, hidden_dim=128):
        super(UpdateBlock, self).__init__()
        # Input: projected corr features + flow (2ch) -> project flow to flow_dim
        self.flow_proj = nn.Conv2d(2, flow_dim, kernel_size=1, bias=False)
        self.gru = nn.GRUCell(input_size=corr_dim + flow_dim,
                              hidden_size=hidden_dim)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, kernel_size=3, stride=1, padding=1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect'),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect'),
        )

    def forward(self, corr_features, flow, hidden):
        B, _, H, W = flow.shape
        # Project flow to flow_dim channels
        flow_proj = self.flow_proj(flow)  # [B, flow_dim, H, W]
        # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C] -> [B*H*W, C]
        corr_flat = corr_features.view(B, -1, H * W).permute(0, 2, 1).reshape(-1, corr_features.shape[1])
        flow_flat = flow_proj.view(B, -1, H * W).permute(0, 2, 1).reshape(-1, flow_proj.shape[1])
        hidden_flat = hidden.view(B, -1, H * W).permute(0, 2, 1).reshape(-1, hidden.shape[1])

        inp = torch.cat([corr_flat, flow_flat], dim=-1)  # [B*H*W, corr_dim+flow_dim]
        hidden_new = self.gru(inp, hidden_flat)  # [B*H*W, hidden_dim]
        hidden_new_spatial = hidden_new.view(B, H * W, -1).permute(0, 2, 1).view(B, -1, H, W)

        delta_flow = self.flow_head(hidden_new_spatial)
        mask = self.mask_head(hidden_new_spatial)

        return delta_flow, mask, hidden_new_spatial


class AlignmentNet(nn.Module):
    """Recurrent Alignment Network based on RAFT with RFDP and DLO.

    Three stages: feature extraction (RFDP), visual similarity computation, iterative update (DLO).
    """
    def __init__(self, in_channels=3, hidden_dim=64, num_levels=4,
                 num_iters=6, corr_dim=256):
        super(AlignmentNet, self).__init__()
        self.num_iters = num_iters

        # Feature extraction with RFDP
        self.feat_encoder = FeatureEncoder(in_channels, hidden_dim, num_levels)

        # Context encoder for reference frame
        self.context_encoder = ContextEncoder(in_channels, hidden_dim, corr_dim)

        # Correlation pyramid
        self.corr_pyramid = CorrelationPyramid(num_levels)

        # DLO
        feat_dim = hidden_dim * (2 ** (num_levels - 1))  # deepest feature dim
        self.dlo = DLO(feat_dim, num_levels, out_dim=corr_dim)

        # Project DLO output to corr_dim
        self.corr_proj = nn.Sequential(
            nn.Conv2d(num_levels * corr_dim, corr_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Update block
        self.update_block = UpdateBlock(
            corr_dim=corr_dim,
            flow_dim=corr_dim,
            hidden_dim=corr_dim
        )

    def initialize_flow(self, img):
        """Initialize zero optical flow field."""
        B, _, H, W = img.shape
        flow = torch.zeros(B, 2, H, W, device=img.device)
        return flow

    def forward(self, ref_img, src_imgs):
        """Align source images to reference image.

        Args:
            ref_img: Reference image [B, C, H, W] (first frame of focal stack)
            src_imgs: Source images [B, N, C, H, W] (remaining frames)

        Returns:
            aligned_imgs: Aligned images [B, N, C, H, W]
            flows: Optical flow fields [B, N, 2, H, W]
        """
        B, N, C, H, W = src_imgs.shape

        # Extract features for reference
        ref_feats = self.feat_encoder(ref_img)
        ref_feat_deep = ref_feats[-1]

        # Context features from reference
        context = self.context_encoder(ref_img)

        aligned_imgs = []
        flows = []

        for i in range(N):
            src_img = src_imgs[:, i]  # [B, C, H, W]
            src_feats = self.feat_encoder(src_img)
            src_feat_deep = src_feats[-1]

            # Compute correlation pyramid
            corr_pyramid = self.corr_pyramid(ref_feat_deep, src_feat_deep)

            # Initialize flow at feature resolution (not image resolution)
            feat_H, feat_W = ref_feat_deep.shape[2], ref_feat_deep.shape[3]
            flow = torch.zeros(B, 2, feat_H, feat_W, device=ref_img.device)
            hidden = torch.zeros(B, context.shape[1], feat_H, feat_W, device=ref_img.device)

            # Iterative update at feature resolution
            for _ in range(self.num_iters):
                # DLO lookup
                corr_features = self.dlo(corr_pyramid, flow)
                # Project to fixed dim
                corr_features = self.corr_proj(corr_features)

                # Update
                delta_flow, mask, hidden = self.update_block(corr_features, flow, hidden)
                flow = flow + delta_flow

            # Upsample flow to image resolution for warping
            flow_up = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
            # Scale flow values to match image resolution
            flow_up = flow_up * (H / feat_H)

            # Warp source image using estimated flow
            warped = self.warp(src_img, flow_up)
            aligned_imgs.append(warped)
            flows.append(flow_up)

        aligned_imgs = torch.stack(aligned_imgs, dim=1)  # [B, N, C, H, W]
        flows = torch.stack(flows, dim=1)  # [B, N, 2, H, W]

        return aligned_imgs, flows

    @staticmethod
    def warp(img, flow):
        """Warp image using optical flow via grid_sample."""
        B, C, H, W = img.shape
        # Create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=img.device, dtype=img.dtype),
            torch.arange(0, W, device=img.device, dtype=img.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # Add flow to grid
        flow_perm = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        grid_new = grid + flow_perm

        # Normalize to [-1, 1]
        grid_new[:, :, :, 0] = 2.0 * grid_new[:, :, :, 0] / max(W - 1, 1) - 1.0
        grid_new[:, :, :, 1] = 2.0 * grid_new[:, :, :, 1] / max(H - 1, 1) - 1.0

        warped = F.grid_sample(img, grid_new, mode='bilinear', padding_mode='zeros',
                               align_corners=False)
        return warped
