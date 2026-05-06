import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 conv layers with BN and LeakyReLU, as described in the paper."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        return x


class WFD(nn.Module):
    """Weighted Feature Downsampling - learned strided convolution for downsampling."""
    def __init__(self, in_channels, out_channels):
        super(WFD, self).__init__()
        self.wfd = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.wfd(x)


class BasicBlock(nn.Module):
    """Residual basic block: two 3x3 convs with skip connection."""
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.lrelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.lrelu(out + residual)
        return out


class UpSample(nn.Module):
    """Upsampling via transposed convolution."""
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class DownSampleBlock(nn.Module):
    """Downsampling block: BasicBlock + WFD."""
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.block = BasicBlock(in_channels)
        self.down = WFD(in_channels, out_channels)

    def forward(self, x):
        x = self.block(x)
        x = self.down(x)
        return x


class ConvRelu(nn.Module):
    """Single 3x3 conv + BN + LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
