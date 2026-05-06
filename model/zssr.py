import torch
import torch.nn as nn
import torch.nn.functional as F


class ZSSRNet(nn.Module):
    """Zero-Shot Super-Resolution network.

    A small 8-layer CNN that learns to super-resolve a single image using
    internal dataset statistics (local self-similarity). No external training data needed.
    """
    def __init__(self, in_channels=3, num_features=64, num_blocks=8):
        super(ZSSRNet, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Middle layers
        for _ in range(num_blocks - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Last layer
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class ZSSR:
    """Zero-Shot Super-Resolution algorithm.

    Uses internal patch recurrence within a single image to train a lightweight
    super-resolution network on the fly. The process:
    1. Downscale the input image to create LR-HR pairs
    2. Train the small network on these pairs
    3. Apply the trained network to the original image

    This is applied to both aligned focal stack images and predicted focus maps
    before they enter the Fusion-Net.
    """
    def __init__(self, scale_factor=2.0, num_features=64, num_blocks=8,
                 num_iters=1000, lr=0.001, device='cuda'):
        self.scale_factor = scale_factor
        self.num_iters = num_iters
        self.lr = lr
        self.device = device
        self.num_features = num_features
        self.num_blocks = num_blocks

    def create_lr_hr_pair(self, hr_img):
        """Create LR-HR training pair by downsampling."""
        lr_img = F.interpolate(hr_img, scale_factor=1.0 / self.scale_factor,
                               mode='bicubic', align_corners=False)
        return lr_img, hr_img

    def train_on_image(self, img):
        """Train ZSSR network for a single image.

        Args:
            img: Input image tensor [B, C, H, W]
        Returns:
            Super-resolved image [B, C, H*scale, W*scale]
        """
        B, C, H, W = img.shape

        # Create internal training pair
        lr_img, hr_img = self.create_lr_hr_pair(img)

        # Initialize small network
        net = ZSSRNet(C, self.num_features, self.num_blocks).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.L1Loss()

        # Train on the internal pair
        net.train()
        for iteration in range(self.num_iters):
            optimizer.zero_grad()
            # Upscale LR to HR size
            lr_upscaled = F.interpolate(lr_img, size=(H, W), mode='bicubic', align_corners=False)
            sr_output = net(lr_upscaled)
            loss = criterion(sr_output, hr_img)
            loss.backward()
            optimizer.step()

        # Apply to original image for super-resolution
        net.eval()
        with torch.no_grad():
            # Upscale original image
            img_upscaled = F.interpolate(img, scale_factor=self.scale_factor,
                                         mode='bicubic', align_corners=False)
            sr_result = net(img_upscaled)

        return sr_result

    def __call__(self, img):
        """Apply ZSSR to an image tensor."""
        return self.train_on_image(img)


def apply_zssr_batch(img_stack, focus_maps, scale_factor=2.0, num_iters=500,
                     device='cuda'):
    """Apply ZSSR to a batch of focal stack images and focus maps.

    Args:
        img_stack: [B, N*C, H, W] focal stack images
        focus_maps: [B, N, H, W] focus maps
        scale_factor: SR scale factor
        num_iters: Number of ZSSR training iterations
        device: Device

    Returns:
        sr_imgs: [B, N*C, H*scale, W*scale] super-resolved images
        sr_focus: [B, N, H*scale, W*scale] super-resolved focus maps
    """
    B = img_stack.shape[0]

    # Process images
    sr_imgs_list = []
    for b in range(B):
        zssr = ZSSR(scale_factor=scale_factor, num_iters=num_iters, device=device)
        sr_img = zssr.train_on_image(img_stack[b:b+1])
        sr_imgs_list.append(sr_img)
    sr_imgs = torch.cat(sr_imgs_list, dim=0)

    # Process focus maps
    sr_focus_list = []
    for b in range(B):
        zssr = ZSSR(scale_factor=scale_factor, num_iters=num_iters,
                     num_features=32, num_blocks=6, device=device)
        sr_fm = zssr.train_on_image(focus_maps[b:b+1])
        sr_focus_list.append(sr_fm)
    sr_focus = torch.cat(sr_focus_list, dim=0)

    return sr_imgs, sr_focus
